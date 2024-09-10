import torch
from loguru import logger
from chai.model import ChaiOne

# Set up model parameters
dim_single = 128
dim_pairwise = 128
dim_msa = 128
dim_msa_input = (
    128  # This should match the last dimension of your msa tensor
)
dim_additional_msa_feats = 2

# Initialize the model
logger.info("Initializing ChaiOne model")
model = ChaiOne(
    dim_single=dim_single,
    dim_pairwise=dim_pairwise,
    msa_depth=4,
    dim_msa=dim_msa,
    dim_msa_input=dim_msa_input,
    dim_additional_msa_feats=dim_additional_msa_feats,
    # outer_product_mean_dim_hidden = 32,
    # msa_pwa_dropout_row_prob = 0.15,
    msa_pwa_heads=8,
    msa_pwa_dim_head=32,
    layerscale_output=False,
    # attention kwargs
    heads=8,
    window_size=120,
    num_memory_kv=0,
    attn_layers=48,
)

# Create dummy input tensors
batch_size = 1
seq_length = 100
num_msa = 4

logger.info(
    f"Creating input tensors with shape: batch_size={batch_size}, seq_length={seq_length}, num_msa={num_msa}"
)
single_repr = torch.randn(batch_size, seq_length, dim_single)
pairwise_repr = torch.randn(
    batch_size, seq_length, seq_length, dim_pairwise
)
msa = torch.randn(batch_size, num_msa, seq_length, dim_msa_input)

# Forward pass
logger.info("Performing forward pass")
output = model(
    single_repr=single_repr,
    pairwise_repr=pairwise_repr,
    msa=msa,
    # mask=mask,
    # msa_mask=msa_mask,
    # attn_bias=attn_bias,
)

logger.info(f"Output shape: {output.shape}")
