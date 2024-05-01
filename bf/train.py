import torch

from diffusers import DiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers import logging as diffusers_logging

from loader import DiffusionDBLoader
import constants as constants
from trainer import BoltFlowTrainer

# pretrained model urls
TEACHER_URL = 'Lykon/dreamshaper-8'
STUDENT_URL = 'Lykon/dreamshaper-8'

# which data split to train on
DATA_SPLIT = 'small'

# BoltFlow trainer hyperparameters
TRAIN_CONFIG = {
    "dtype": torch.bfloat16,
    "lr": 3e-6,
    "bs": 8,
    "grad_accum_steps": 1,
    "num_steps": 8000,
    "warmup_steps": 1000,
    "eval_freq": 500,
    "checkpoint_freq": 4000,
    "guidance_scale": 7.5,
    "constant_noise": False,
    "example_prompts": [
        "a cyberpunk woman riding a red motorcycle",
        "a photo of a cat",
        "a spaceship landing in a city",
        "a jungle filled with robots",
        "a wizard casting an ice spell",
        "a car cruising down the coast at sunset"
    ],
}

# whether to train with debugging
DEBUG = False

# name of the run
NAME = 'boltflow-8-const'


def main():
    
    # disable diffusers warnings
    diffusers_logging.set_verbosity_error()

    print("Loading models...")
    
    # pipe and teacher
    scheduler = DDIMScheduler.from_pretrained(TEACHER_URL, subfolder='scheduler')
    pipe = DiffusionPipeline.from_pretrained(TEACHER_URL, scheduler=scheduler)
    pipe = pipe.to(constants.DEVICE)
    torch.compile(pipe, mode="reduce-overhead", fullgraph=True)
    teacher_unet = pipe.unet

    # student
    student_unet = UNet2DConditionModel.from_pretrained(STUDENT_URL, subfolder="unet")
    student_unet = student_unet.to(constants.DEVICE)
    torch.compile(student_unet, mode="reduce-overhead", fullgraph=True)

    print("Loading data...")
    loader = DiffusionDBLoader(DATA_SPLIT, debug=DEBUG)

    print("Train!")
    trainer = BoltFlowTrainer(
        NAME,
        debug=DEBUG,
        **TRAIN_CONFIG
    )
    trainer.train(
        pipe,
        teacher_unet,
        student_unet,
        loader
    )


if __name__ == "__main__":
        main()
