import torch.nn as nn

from .ARDecoder import ARDecoder
from .conv import BLConv
from .lstm import LSTM
from .transformer import Transformer


def build_encoder(in_chan: int, arch: str, len_seq: int = 0) -> nn.Module:
    match arch:
        case 'blconv_b':
            # The shared 1D CNN encoder used throughout (REWI-style, ~2.46 M params).
            return BLConv(in_chan)
        case _:
            raise ValueError(f"Unknown encoder arch: {arch}")


def build_decoder(
    dim_in: int,
    num_cls: int,
    arch: str,
    len_seq: int = 0,
    *,
    use_gated_attention: bool = False,
    gating_type: str = "elementwise",
) -> nn.Module:
    match arch:
        case 'bilstm_wide':
            # REWI BiLSTM-CTC baseline decoder.
            return LSTM(dim_in, num_cls, hidden_size=164, num_layers=3, r_drop=0.2)

        case 'transformer_ctc':
            # Parameter-matched Transformer-CTC variant: a self-attention encoder
            # with a CTC head, same d_model / depth / heads as the AR decoder but a
            # wider FFN (1472 vs 896) to absorb the budget the AR decoder spends on
            # cross-attention. With the param-free sinusoidal PE the total lands at
            # 4.649 M (+0.27 % vs REWI 4.637 M).
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=256, nhead=4, num_layers=2, dim_ff=1472,
                               p_drop=0.1, apply_softmax=False)

        case 'ar_transformer':
            # HWRFormer's AR Transformer decoder (cross-attention), parameter-matched
            # to REWI's CTC baseline (~4.64 M total with blconv_b).
            return ARDecoder(vocab_size=num_cls, d_model=256, nhead=4, layers=2,
                             dim_ff=896, pdrop=0.1, use_gated_attention=use_gated_attention,
                             gating_type=gating_type)

        case _:
            raise ValueError(f"Unknown decoder arch: {arch}")
