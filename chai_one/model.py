from loguru import logger
from alphafold3_pytorch import MSAModule, AttentionPairBias
from typing import Optional, Dict, Any
import torch.nn as nn
from torch import Tensor


class ChaiOne(nn.Module):
    def __init__(
        self,
        dim_single: int = 384,
        dim_pairwise: int = 128,
        msa_depth: int = 4,
        dim_msa: int = 64,
        dim_msa_input: Optional[int] = None,
        dim_additional_msa_feats: int = 0,
        outer_product_mean_dim_hidden: int = 32,
        msa_pwa_dropout_row_prob: float = 0.15,
        msa_pwa_heads: int = 8,
        msa_pwa_dim_head: int = 32,
        checkpoint: bool = False,
        pairwise_block_kwargs: Dict[str, Any] = dict(),
        max_num_msa: int | None = None,
        layerscale_output: bool = False,
        # attention kwargs
        heads: int = 8,
        window_size: Optional[int] = None,
        num_memory_kv: int = 120,
        attn_layers: int = 48,
    ):
        super().__init__()
        self.msa_module = MSAModule(
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            depth=msa_depth,
            dim_msa=dim_msa,
            dim_msa_input=dim_msa_input,
            dim_additional_msa_feats=dim_additional_msa_feats,
            outer_product_mean_dim_hidden=outer_product_mean_dim_hidden,
            msa_pwa_dropout_row_prob=msa_pwa_dropout_row_prob,
            msa_pwa_heads=msa_pwa_heads,
            msa_pwa_dim_head=msa_pwa_dim_head,
            checkpoint=checkpoint,
            pairwise_block_kwargs=pairwise_block_kwargs,
            max_num_msa=max_num_msa,
            layerscale_output=layerscale_output,
        )

        # Attention
        self.attn_layers = nn.ModuleList(
            [
                AttentionPairBias(
                    heads=heads,
                    dim_pairwise=dim_pairwise,
                    window_size=window_size,
                    num_memory_kv=num_memory_kv,
                )
                for _ in range(attn_layers)
            ]
        )

    def forward(
        self,
        single_repr: Tensor,
        pairwise_repr: Tensor,
        msa: Tensor,
        mask: Tensor | None = None,
        msa_mask: Tensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        msa_feats = self.msa_module(
            single_repr=single_repr,
            pairwise_repr=pairwise_repr,
            msa=msa,
            # mask=mask,
            # msa_mask=msa_mask,
        )

        logger.info(f"MSA feats shape: {msa_feats.shape}")

        # Attention Pairwise 48 Layers
        # single_repr: Float['b n ds'],
        # pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        # attn_bias: Float['b n n'] | Float['b nw w (w*2)'] | None = None,
        for attn in self.attn_layers:
            msa_feats = attn(
                single_repr=single_repr,
                pairwise_repr=msa_feats,
                attn_bias=attn_bias,
            )
            
        logger.info(f"MSA feats shape: {msa_feats.shape}")

        return msa_feats
