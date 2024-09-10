[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# Chai-1

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

An free and open source community implementation of Chai-1 in PyTorch. [Paper is here](https://chaiassets.com/chai-1/paper/technical_report_v1.pdf)

Join our discord to help us implement this paper!


## Installation

```bash
pip3 install chai-one
```

## Usage

```python

######### example.py
import torch
from loguru import logger
from chai_one.model import ChaiOne

# Set up model parameters
dim_single = 128
dim_pairwise = 128
dim_msa = 128
dim_msa_input = 134  # Adjusted to match the expected input dimension
dim_additional_msa_feats = 2
window_size = 25

# Initialize the model
logger.info("Initializing ChaiOne model")
model = ChaiOne(
    dim_single=dim_single,
    dim_pairwise=dim_pairwise,
    msa_depth=4,
    dim_msa=dim_msa,
    dim_msa_input=dim_msa_input,  # Set to 134
    dim_additional_msa_feats=0,
    msa_pwa_heads=8,
    msa_pwa_dim_head=32,
    layerscale_output=False,
    heads=8,
    window_size=window_size,
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

# Create msa tensor with matching input size for msa_init_proj (134 features)
msa = torch.randn(
    batch_size, num_msa, seq_length, dim_msa_input
)  # Adjusted to 134

# Forward pass
logger.info("Performing forward pass")
output = model(
    single_repr=single_repr,
    pairwise_repr=pairwise_repr,
    msa=msa,
)

logger.info(f"Output shape: {output.shape}")

```



# License
MIT
