import torch

# best device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# local data path
LOCAL_DATA_DIR = "./local_data"

# huggingface login id
HF_ID = "aklein4"

# shape of the stable diffusion latent space, with batch size
LATENT_SHAPE = lambda bs: (bs, 4, 64, 64)
