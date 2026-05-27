"""
Main entry point for IMU Handwriting Recognition (HWR) training and evaluation.

Supports three training regimes:
- CTC (recurrent BiLSTM-CTC baseline or Transformer-CTC)
- AR (autoregressive Transformer decoder, optionally with SDPA output gating)
- Hybrid CTC-AR (auxiliary CTC head on the shared encoder)

Usage:
    python main.py -c configs/<experiment>.yaml      # training
    python main.py -c configs/<experiment>.yaml      # evaluation (set test: true)
"""

import argparse
import os
import warnings

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from hwrformer.ctc_decoder import BestPath
from hwrformer.dataset import HRDataset
from hwrformer.dataset.utils import fn_collate
from hwrformer.loss import CTCLoss
from hwrformer.manager import RunManager
from hwrformer.model import BaseModel, DualHeadModel
from hwrformer.utils import seed_everything, seed_worker

from hwrformer.training import (
    train_one_epoch,
    train_one_epoch_hybrid,
    test,
    test_hybrid,
    maybe_log_trainability,
    maybe_log_optimizer_coverage,
    set_decoder_frozen,
    log_decoder_pretrain_load,
)

warnings.filterwarnings('ignore', category=UserWarning)


def setup_tokenizer(cfgs: argparse.Namespace):
    """Set up character-level vocabulary and special-token IDs.

    Returns:
        tok: always None (character mode).
        vocab_dec: decoder vocabulary size.
        PAD_ID, BOS_ID, EOS_ID: special token IDs.
    """
    AR_MODE = cfgs.arch_de in {
        "ar_transformer"
    }

    base = len(cfgs.categories)
    PAD_ID, BOS_ID, EOS_ID = base, base + 1, base + 2
    vocab_dec = base + (3 if AR_MODE else 0)
    return None, vocab_dec, PAD_ID, BOS_ID, EOS_ID


def build_model(cfgs: argparse.Namespace, manager: RunManager):
    """Build a CTC / AR / hybrid model from the configuration."""
    dual = getattr(cfgs, "dual_head", {}) or {}
    dual_enabled = bool(dual.get("enabled", False))

    if dual_enabled:
        vocab_ctc = int(len(cfgs.categories))
        arch_ctc = str(dual.get("arch_ctc", "linear"))
        tie_cfg = dual.get("tie", {}) or {}

        model = DualHeadModel(
            arch_en=cfgs.arch_en,
            arch_ar=cfgs.arch_de,
            arch_ctc=arch_ctc,
            in_chan=cfgs.num_channel,
            vocab_ar=cfgs.vocab_dec,
            vocab_ctc=vocab_ctc,
            len_seq=cfgs.len_seq,
            use_gated_attention=getattr(cfgs, "use_gated_attention", False),
            gating_type=getattr(cfgs, "gating_type", "elementwise"),
            pad_id=getattr(cfgs, "PAD_ID", None),
            bos_id=getattr(cfgs, "BOS_ID", None),
            eos_id=getattr(cfgs, "EOS_ID", None),
            tie_cfg=tie_cfg,
            dual_cfg=dual,
        ).to(cfgs.device)
        cfgs.DUAL_HEAD = True
        cfgs.vocab_ctc = vocab_ctc
        cfgs.dual_head_arch_ctc = arch_ctc
    else:
        dec_ctc_cfg = getattr(cfgs, "decoder_side_ctc", {}) or {}
        vocab_ctc = int(len(cfgs.categories))

        model = BaseModel(
            cfgs.arch_en,
            cfgs.arch_de,
            cfgs.num_channel,
            cfgs.vocab_dec,
            cfgs.len_seq,
            use_gated_attention=getattr(cfgs, "use_gated_attention", False),
            gating_type=getattr(cfgs, "gating_type", "elementwise"),
            vocab_ctc=vocab_ctc,
            pad_id=getattr(cfgs, "PAD_ID", None),
            decoder_side_ctc_cfg=dec_ctc_cfg,
        ).to(cfgs.device)
        cfgs.DUAL_HEAD = False
        cfgs.vocab_ctc = vocab_ctc

    # Optional pretrained decoder initialization
    pretrained_dec_ckpt = getattr(cfgs, "pretrained_decoder_checkpoint", None)
    if pretrained_dec_ckpt:
        ckp = torch.load(pretrained_dec_ckpt, map_location="cpu", weights_only=False)
        state = ckp.get("model", ckp)
        log_decoder_pretrain_load(model, ckpt_path=str(pretrained_dec_ckpt), state=state)
        maybe_log_trainability(manager, model, epoch=0, where="after_pretrained_decoder_load")

    return model


def build_dataloaders(cfgs: argparse.Namespace, model, tok):
    """Build training and test dataloaders (character mode, standard collation)."""
    dataset_test = HRDataset(
        os.path.join(cfgs.dir_dataset, 'val.json'),
        cfgs.categories,
        model.ratio_ds,
        cfgs.idx_fold,
        cfgs.len_seq,
        cache=cfgs.cache,
    )
    dataset_test.ignore_unknown_chars = bool(getattr(cfgs, 'ignore_unknown_chars', False))
    dataset_test.return_raw_label = bool(getattr(cfgs, 'return_raw_label', False))

    dataloader_test = DataLoader(
        dataset_test,
        cfgs.size_batch,
        num_workers=cfgs.num_worker,
        collate_fn=fn_collate,
    )

    dataloader_train = None
    if not cfgs.test:
        dataset_train = HRDataset(
            os.path.join(cfgs.dir_dataset, 'train.json'),
            cfgs.categories,
            model.ratio_ds,
            cfgs.idx_fold,
            cfgs.len_seq,
            cfgs.aug,
            cfgs.cache,
        )
        dataset_train.tokenizer = tok
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=cfgs.size_batch,
            shuffle=True,
            num_workers=cfgs.num_worker,
            collate_fn=fn_collate,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfgs.seed),
        )

    return dataloader_train, dataloader_test


def build_optimizer_and_scheduler(cfgs, model, dataloader_train):
    """Standard AdamW optimizer + linear-warmup/cosine-decay scheduler."""
    optimizer = torch.optim.AdamW(model.parameters(), cfgs.lr)

    use_amp = bool(getattr(cfgs, "use_amp", True))
    scaler = GradScaler(enabled=use_amp, init_scale=2048.0)
    logger.info("[AMP] {}", "enabled (init_scale=2048)" if use_amp else "disabled")

    if dataloader_train is not None:
        lr_scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, 0.01, total_iters=len(dataloader_train) * cfgs.epoch_warmup),
                CosineAnnealingLR(optimizer, len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup)),
            ],
            [len(dataloader_train) * cfgs.epoch_warmup],
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    return optimizer, scaler, lr_scheduler


def _migrate_legacy_mha_keys(state_dict: dict) -> dict:
    """Migrate old separate-projection MHA keys to nn.MultiheadAttention format.

    Old custom GatedMultiheadAttention stored q/k/v/out projections separately;
    the current code wraps nn.MultiheadAttention (in_proj_weight/out_proj). This
    concatenates the legacy q/k/v tensors so old checkpoints still load.
    """
    import re

    new_sd = {}
    consumed = set()

    old_prefixes = set()
    for k in state_dict:
        m = re.match(r"^(.+\.(self_attn|multihead_attn))\.q_proj\.weight$", k)
        if m:
            old_prefixes.add(m.group(1))

    if old_prefixes:
        logger.info("[MHAMigrate] Found {} attention modules with legacy keys", len(old_prefixes))

    for prefix in old_prefixes:
        q_w = state_dict.get(f"{prefix}.q_proj.weight")
        k_w = state_dict.get(f"{prefix}.k_proj.weight")
        v_w = state_dict.get(f"{prefix}.v_proj.weight")
        q_b = state_dict.get(f"{prefix}.q_proj.bias")
        k_b = state_dict.get(f"{prefix}.k_proj.bias")
        v_b = state_dict.get(f"{prefix}.v_proj.bias")
        o_w = state_dict.get(f"{prefix}.out_proj.weight")
        o_b = state_dict.get(f"{prefix}.out_proj.bias")

        if q_w is not None and k_w is not None and v_w is not None:
            new_sd[f"{prefix}.mha.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
            consumed.update([f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"])
            if q_b is not None and k_b is not None and v_b is not None:
                new_sd[f"{prefix}.mha.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
                consumed.update([f"{prefix}.q_proj.bias", f"{prefix}.k_proj.bias", f"{prefix}.v_proj.bias"])
        if o_w is not None:
            new_sd[f"{prefix}.mha.out_proj.weight"] = o_w
            consumed.add(f"{prefix}.out_proj.weight")
        if o_b is not None:
            new_sd[f"{prefix}.mha.out_proj.bias"] = o_b
            consumed.add(f"{prefix}.out_proj.bias")

    for k, v in state_dict.items():
        if k not in consumed:
            new_sd[k] = v

    return new_sd


def load_checkpoint(cfgs, model, optimizer, lr_scheduler, manager, dataloader_train, epoch_start):
    """Load a checkpoint and handle resume / freeze / finetune modes."""
    if not cfgs.checkpoint:
        return epoch_start, optimizer, lr_scheduler

    map_loc = torch.device("cpu") if str(cfgs.device) == "cpu" or not torch.cuda.is_available() else None
    ckp = torch.load(cfgs.checkpoint, map_location=map_loc, weights_only=False)

    ckp["model"] = _migrate_legacy_mha_keys(ckp["model"])

    # Filter out keys with shape mismatches (e.g. CTC head when categories change)
    model_sd = model.state_dict()
    filtered_sd = {}
    shape_skipped = []
    for k, v in ckp["model"].items():
        if k in model_sd and model_sd[k].shape != v.shape:
            shape_skipped.append(k)
        else:
            filtered_sd[k] = v
    if shape_skipped:
        logger.warning("load_state_dict: skipped shape-mismatched keys: {}", shape_skipped)
    res = model.load_state_dict(filtered_sd, strict=False)
    logger.warning("load_state_dict strict=False | missing={} unexpected={}", res.missing_keys, res.unexpected_keys)

    if not cfgs.test:
        if 'epoch' in ckp.keys():  # Resume
            epoch_start = ckp['epoch'] + 1
            try:
                optimizer.load_state_dict(ckp['optimizer'])
                lr_scheduler.load_state_dict(ckp['lr_scheduler'])
                logger.info("[Resume] epoch_start={} | checkpoint={}", epoch_start, cfgs.checkpoint)
            except (ValueError, KeyError) as e:
                logger.warning(
                    "[Resume] optimizer/scheduler load failed ({}); "
                    "warm-restarting from epoch {} with fresh optimizer",
                    e, epoch_start,
                )
            maybe_log_optimizer_coverage(manager, optimizer, model, epoch=epoch_start, where="after_resume")

        elif cfgs.freeze:  # Freeze encoder
            for params in model.encoder.parameters():
                params.requires_grad = False
            logger.info("[Freeze] encoder frozen | checkpoint={}", cfgs.checkpoint)
            maybe_log_optimizer_coverage(manager, optimizer, model, epoch=epoch_start, where="after_freeze")

        else:  # Finetune with discriminative LR
            optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': cfgs.lr * 0.1},
                {'params': model.decoder.parameters(), 'lr': cfgs.lr},
            ])
            lr_scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(optimizer, 0.01, total_iters=len(dataloader_train) * cfgs.epoch_warmup),
                    CosineAnnealingLR(optimizer, len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup)),
                ],
                [len(dataloader_train) * cfgs.epoch_warmup],
            )
            maybe_log_optimizer_coverage(manager, optimizer, model, epoch=epoch_start, where="after_finetune")

        maybe_log_trainability(manager, model, epoch=epoch_start, where="after_checkpoint_load")

    logger.info(f'Load checkpoint from {cfgs.checkpoint}')
    return epoch_start, optimizer, lr_scheduler


def run_training_loop(
    cfgs, model, optimizer, scaler, lr_scheduler, manager,
    dataloader_train, dataloader_test, ctc_decoder, tok,
    epoch_start, AR_MODE,
):
    """Train/evaluate over epochs for CTC / AR / hybrid regimes."""
    dual = getattr(cfgs, "dual_head", {}) or {}
    dual_enabled = bool(dual.get("enabled", False)) and bool(AR_MODE)

    fn_loss = (
        nn.CrossEntropyLoss(ignore_index=cfgs.PAD_ID, label_smoothing=0.1)
        if AR_MODE else CTCLoss()
    )

    fn_loss_ar = fn_loss_ctc = None
    lambda_ar, lambda_ctc = 1.0, 0.0
    lambda_ctc_schedule = None
    loss_balance_mode = "sum"
    if dual_enabled:
        fn_loss_ar = nn.CrossEntropyLoss(ignore_index=cfgs.PAD_ID, label_smoothing=0.1)
        fn_loss_ctc = CTCLoss()
        lambda_ar = float(dual.get("lambda_ar", 1.0))
        lambda_ctc = float(dual.get("lambda_ctc", 0.3))
        lambda_ctc_schedule = dual.get("lambda_ctc_schedule", None)
        loss_balance_mode = str(dual.get("loss_balance", "sum"))

    for e in range(epoch_start, cfgs.epoch):
        if cfgs.test:
            _run_test_epoch(cfgs, model, fn_loss, manager, dataloader_test, ctc_decoder, tok)
            break

        # Optional decoder freezing (AR ablation)
        if hasattr(model, "decoder"):
            freeze_dec_epochs = int(getattr(cfgs, "freeze_decoder_epochs", 0) or 0)
            if freeze_dec_epochs > 0:
                frozen = int(e) < freeze_dec_epochs
                set_decoder_frozen(manager, model, frozen=frozen, epoch=int(e), where="main_loop")
                maybe_log_optimizer_coverage(manager, optimizer, model, epoch=e, where="after_decoder_freeze_toggle")

        # Train
        if dual_enabled:
            train_one_epoch_hybrid(
                dataloader_train, model, fn_loss_ar, fn_loss_ctc,
                lambda_ar=lambda_ar, lambda_ctc=lambda_ctc,
                lambda_ctc_schedule=lambda_ctc_schedule, loss_balance_mode=loss_balance_mode,
                optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler, man=manager, epoch=e,
            )
        else:
            train_one_epoch(dataloader_train, model, fn_loss, optimizer, scaler, lr_scheduler, manager, e)

        # Evaluate
        if dual_enabled:
            test_hybrid(
                dataloader_test, model, fn_loss_ar, fn_loss_ctc,
                lambda_ar=lambda_ar, lambda_ctc=lambda_ctc,
                lambda_ctc_schedule=lambda_ctc_schedule, loss_balance_mode=loss_balance_mode,
                man=manager, ctc_decoder=ctc_decoder, epoch=e, tokenizer=tok,
            )
        else:
            test(dataloader_test, model, fn_loss, manager, ctc_decoder, e, tokenizer=tok)

        _save_checkpoints(cfgs, model, optimizer, lr_scheduler, manager, e)

    if not cfgs.test:
        manager.summarize_evaluation()


def _run_test_epoch(cfgs, model, fn_loss, manager, dataloader_test, ctc_decoder, tok):
    """Run a single test/evaluation epoch (CTC / AR / hybrid)."""
    dual = getattr(cfgs, "dual_head", {}) or {}
    dual_enabled = bool(getattr(cfgs, "DUAL_HEAD", False)) or bool(dual.get("enabled", False))

    if dual_enabled:
        fn_loss_ar = nn.CrossEntropyLoss(ignore_index=cfgs.PAD_ID, label_smoothing=0.1)
        fn_loss_ctc = CTCLoss()
        test_hybrid(
            dataloader_test, model, fn_loss_ar, fn_loss_ctc,
            lambda_ar=float(dual.get("lambda_ar", 1.0)),
            lambda_ctc=float(dual.get("lambda_ctc", 0.3)),
            lambda_ctc_schedule=dual.get("lambda_ctc_schedule", None),
            loss_balance_mode=str(dual.get("loss_balance", "sum")),
            man=manager, ctc_decoder=ctc_decoder, epoch=0, tokenizer=tok, force_eval=True,
        )
    else:
        test(dataloader_test, model, fn_loss, manager, ctc_decoder, 0, tokenizer=tok, force_eval=True)
    manager.summarize_evaluation()


def _save_checkpoints(cfgs, model, optimizer, lr_scheduler, manager, e):
    """Save best (per-metric) and last checkpoints."""
    def _maybe_save_best_from_bestdict(best_dict: dict, *, prefix: str, also_update_primary: bool):
        if not isinstance(best_dict, dict):
            return
        if "character_error_rate" in best_dict and int(best_dict["character_error_rate"][0]) == int(e):
            manager.save_checkpoint(
                model.state_dict(), optimizer.state_dict(),
                lr_scheduler.state_dict(), filename=f"best_{prefix}_cer.pth"
            )
            if also_update_primary:
                manager.save_checkpoint(
                    model.state_dict(), optimizer.state_dict(),
                    lr_scheduler.state_dict(), filename="best_cer.pth"
                )
        if "word_error_rate" in best_dict and int(best_dict["word_error_rate"][0]) == int(e):
            manager.save_checkpoint(
                model.state_dict(), optimizer.state_dict(),
                lr_scheduler.state_dict(), filename=f"best_{prefix}_wer.pth"
            )
            if also_update_primary:
                manager.save_checkpoint(
                    model.state_dict(), optimizer.state_dict(),
                    lr_scheduler.state_dict(), filename="best_wer.pth"
                )

    if bool(getattr(cfgs, "save_best_only", False)):
        dual = getattr(cfgs, "dual_head", {}) or {}
        dual_enabled = bool(getattr(cfgs, "DUAL_HEAD", False)) or bool(dual.get("enabled", False))

        # Backward-compatible AR best tracking under key='evaluation'.
        manager.summarize_evaluation(key="evaluation")
        best_ar = manager.results.get("best", {})

        best_ctc = {}
        if dual_enabled:
            manager.summarize_evaluation(key="evaluation_ctc")
            best_ctc = manager.results.get("best_evaluation_ctc", {})

        # Choose which head defines best_cer.pth / best_wer.pth.
        primary_raw = dual.get("primary", dual.get("primary_head", None))
        primary = str(primary_raw).lower() if primary_raw is not None else "auto"
        if primary in {"", "none", "null"}:
            primary = "auto"
        if primary == "auto" and dual_enabled:
            try:
                lambda_ar_cfg = float(dual.get("lambda_ar", 1.0))
            except Exception:
                lambda_ar_cfg = 1.0
            try:
                lambda_ctc_cfg = float(dual.get("lambda_ctc", 0.0))
            except Exception:
                lambda_ctc_cfg = 0.0
            sched = dual.get("lambda_ctc_schedule", None) or {}
            if bool(sched.get("enabled", False)) and "max" in sched:
                try:
                    lambda_ctc_ref = float(sched.get("max", lambda_ctc_cfg))
                except Exception:
                    lambda_ctc_ref = lambda_ctc_cfg
            else:
                lambda_ctc_ref = lambda_ctc_cfg
            primary = "ctc" if lambda_ctc_ref >= lambda_ar_cfg else "ar"
        if primary not in {"ar", "ctc"}:
            primary = "ar"

        _maybe_save_best_from_bestdict(best_ar, prefix="ar", also_update_primary=(primary == "ar"))
        if dual_enabled:
            _maybe_save_best_from_bestdict(best_ctc, prefix="ctc", also_update_primary=(primary == "ctc"))

    if not cfgs.test:
        manager.save_checkpoint(
            model.state_dict(), optimizer.state_dict(),
            lr_scheduler.state_dict(), filename="last.pth"
        )


def main(cfgs: argparse.Namespace) -> None:
    """Train or evaluate a CTC / AR / hybrid model from a YAML config."""
    AR_MODE = cfgs.arch_de in {
        "ar_transformer"
    }
    cfgs.AR_MODE = bool(AR_MODE)

    tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID = setup_tokenizer(cfgs)
    cfgs.PAD_ID, cfgs.BOS_ID, cfgs.EOS_ID = PAD_ID, BOS_ID, EOS_ID
    cfgs.vocab_dec = vocab_dec
    cfgs.tokenizer_obj = tok

    dual = getattr(cfgs, "dual_head", {}) or {}
    if AR_MODE and bool(dual.get("enabled", False)):
        cfgs.TRAINING_REGIME = "hybrid_ar_ctc"
    elif AR_MODE:
        cfgs.TRAINING_REGIME = "ar_only"
    else:
        cfgs.TRAINING_REGIME = "ctc_only"

    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    ctc_decoder = BestPath(cfgs.categories)

    model = build_model(cfgs, manager)
    dataloader_train, dataloader_test = build_dataloaders(cfgs, model, tok)
    optimizer, scaler, lr_scheduler = build_optimizer_and_scheduler(cfgs, model, dataloader_train)

    epoch_start = 0
    maybe_log_optimizer_coverage(manager, optimizer, model, epoch=epoch_start, where="after_optimizer_init")

    epoch_start, optimizer, lr_scheduler = load_checkpoint(
        cfgs, model, optimizer, lr_scheduler, manager, dataloader_train, epoch_start
    )

    run_training_loop(
        cfgs, model, optimizer, scaler, lr_scheduler, manager,
        dataloader_train, dataloader_test, ctc_decoder, tok,
        epoch_start, AR_MODE,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='IMU Handwriting Recognition - Training and Evaluation'
    )
    parser.add_argument('-c', '--config', required=True, help='Path to YAML configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfgs = argparse.Namespace(**yaml.safe_load(f))

    main(cfgs)
