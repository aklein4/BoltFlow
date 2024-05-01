from typing import Optional

import torch
import torch.nn.functional as F

import os
import pandas as pd
import yaml
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import huggingface_hub as hf

import constants as constants
from training_utils import decode_image, get_original_sample, boltflow_loss
from loader import DEBUG_PROMPT


class BoltFlowTrainer:

    # file saving locations
    _hyper_file = "hyperparams.yml"
    _plot_file = "progress.png"
    _example_file = "example.png"
    _noise_file = "noise.pt"

    # hyperparameters used
    _hyperparams = [
        "dtype", # dtype to use for mixed precision
        "lr", # learning rate
        "bs", # batch size
        "grad_accum_steps", # number of gradient accumulation steps
        "num_steps", # number of training steps
        "warmup_steps", # number of linear lr warmup steps
        "eval_freq", # frequency to evaluate the model in training steps
        "checkpoint_freq", # frequency to save the model in training steps
        "guidance_scale", # teacher guidance scale
        "constant_noise", # whether to use constant noise for teacher
        "example_prompts", # prompts to use for evaluation
    ]

    def __init__(
        self,
        save_name: str,
        debug: Optional[bool]=False,
        **kwargs
    ):
        """ Trainer for BoltFlow distillation.
        See BoltFLowTrainer._hyperparams for kwargs that must be passed.

        Args:
            save_name (str): Name of the model to save to huggingface. 
            ddebug (bool, optional): Make various changes for debugging. Defaults to False.
            kwargs: hyperparameters for the trainer. 
        """
        self.debug = debug

        # init save locations
        self.save_name = save_name
        self.save_repo = f"{constants.HF_ID}/{save_name}"
        hf.create_repo(
            save_name, private=True, exist_ok=True
        )
        os.makedirs(constants.LOCAL_DATA_DIR, exist_ok=True)

        # set hyperparameters        
        for k in self._hyperparams:
            setattr(self, k, kwargs[k])

        # init progress log
        self.loss_log = []

        # initialize noise
        self.noise = None
        if self.constant_noise:
            self.noise = torch.randn(*constants.LATENT_SHAPE(1)).to(constants.DEVICE)


    @torch.no_grad()
    def save_log(self):
        """ Save the training progress to huggingface.
        Uploads:
         - hyperparameters yaml
         - plot of the loss
         - example images
         - noise tensor (if constant_noise)
        """
        api = hf.HfApi()

        # save hyperparams as csv
        with open(self._hyper_file, 'w') as outfile:
            yaml.dump(
                {k: str(getattr(self, k)) for k in self._hyperparams},
                outfile,
                default_flow_style=False
            )

        # get rolling loss
        df = pd.DataFrame({"loss": self.loss_log})
        roll = df.rolling(window=self.eval_freq, center=False, min_periods=self.eval_freq//2)
        mask = ~np.isnan(roll["loss"].mean())
        steps_to_plot = np.arange(len(roll["loss"].mean()))[mask]

        # save plot of rolling loss
        plt.plot(steps_to_plot, roll["loss"].mean()[mask])
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"BoltFlow Training Progress (Step {len(self.log.loss)})")
        plt.savefig(self._plot_file)
        plt.close()

        # save noise
        if self.constant_noise:
            torch.save(self.noise, self._noise_file)

        # upload files
        files = [self._hyper_file, self._plot_file, self._example_file]
        if self.constant_noise:
            files.append(self._noise_file)
        for file in files:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=self.save_repo,
                repo_type="model"
            )


    @torch.no_grad()
    def save_checkpoint(
        self,
        student_unet,
    ):
        """ Save the student unet checkpoint to huggingface.

        Args:
            student_unet (UNet2DConditionModel): Student unet model to save.
        """
        api = hf.HfApi()

        student_unet.save_pretrained(
            "unet",
            push_to_hub=False,
            # repo_id=self.save_repo
        )

        for file in ["unet"]:
            api.upload_folder(
                    repo_id=self.save_repo,
                    folder_path=file,
                    path_in_repo=file,
                    repo_type="model"
            )


    @torch.no_grad()
    def evaluate(
        self,
        pipe,
        student_unet,
        step: int
    ):
        """ Evaluate the student unet by generating example images.

        Args:
            pipe (StableDiffusionPipeline): Pipeline containing the scheduler and vae.
            student_unet (UNet2DConditionModel): Student unet model to evaluate.
            step (int): Current training step
        """
        student_unet.eval()

        # handle debugging
        if self.debug:
            self.example_prompts = [DEBUG_PROMPT, DEBUG_PROMPT]

        # useful vars
        n_examples = len(self.example_prompts)
        max_t = pipe.scheduler.config.num_train_timesteps - 1

        # vector of timesteps filled with the max timestep
        t_vec = torch.full(
                (n_examples,), max_t,
                dtype=torch.long, device=constants.DEVICE
        )

        # get the prompt embeddings
        prompt_embeds, _ = pipe.encode_prompt(self.example_prompts, constants.DEVICE, 1, False)

        # generate noise deterministically
        generator = torch.Generator().manual_seed(0)
        noise = torch.randn(
            *constants.LATENT_SHAPE(n_examples),
            generator=generator
        ).to(constants.DEVICE)

        # get the model output
        student_output = student_unet(
            noise,
            t_vec,
            prompt_embeds,
        ).sample
        x = get_original_sample(
            pipe.scheduler,
            student_output,
            t_vec,
            noise
        )

        # decode the images
        images = decode_image(pipe, x)

        # plot the images
        fig, ax = plt.subplots(1, n_examples, figsize=(5*n_examples, 5))
        for i in range(n_examples):
            ax[i].imshow(images[i])
            ax[i].axis("off")

        plt.suptitle(f"BoltFlow Examples (Step {step})")
        plt.tight_layout()
        plt.savefig(self._example_file)
        plt.show()
        plt.close()


    def train(
        self,
        pipe,
        teacher_unet,
        student_unet,
        loader
    ):
        """
        Train the student unet using the BoltFlow distillation method.

        Args:
            pipe (StableDiffusionPipeline): Pipeline containing the scheduler and vae.
            teacher_unet (UNet2DConditionModel): Teacher unet model to distill from.
            student_unet (UNet2DConditionModel): Student unet model to train.
            loader (DiffusionDBLoader): Loader for prompts to use in training.
        """

        # useful vars
        num_t = pipe.scheduler.config.num_train_timesteps

        # init scheduler
        pipe.scheduler.set_timesteps(num_t)

        # set up network modes
        student_unet.train()
        teacher_unet.eval()
        pipe.text_encoder.eval()
        pipe.vae.eval()

        # set up network grads
        student_unet.requires_grad_(True)
        teacher_unet.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.vae.requires_grad_(False)

        # init training objs
        optimizer = torch.optim.AdamW(
            student_unet.parameters(),
            lr=self.lr
        )
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        scaler = torch.cuda.amp.GradScaler()

        # create reusable tensors
        with torch.no_grad():
            
            # negative prompt embeds
            negative_embeds, _ = pipe.encode_prompt(
                "", constants.DEVICE, 1, False
            )
            negative_embeds = negative_embeds.expand(self.bs, -1, -1)

            # timestep vector for the student unet
            student_t_vec = torch.full(
                (self.bs,), num_t-1,
                dtype=torch.long, device=constants.DEVICE
            )

        # prepare for loop
        self.loss_log = []
        loader.reset()
        generator = torch.Generator().manual_seed(0)

        # run training loop
        with tqdm(range(self.num_steps)) as pbar:
            for step in pbar:

                # handle coming back from evaluation
                student_unet.train()

                # accumulate loss (loss_accum is for logging)
                loss_accum = 0.0
                for grad_step in range(self.grad_accum_steps):

                    # handle mixed precision
                    enable_autocast = self.dtype != torch.float32
                    with torch.autocast(
                        device_type=str(constants.DEVICE),
                        dtype=(torch.float16 if not enable_autocast else self.dtype),
                        enabled=enable_autocast
                    ):

                        # get the prompts and embeddigns
                        prompts = loader(self.bs)
                        with torch.no_grad():
                            prompt_embeds, _ = pipe.encode_prompt(prompts, constants.DEVICE, 1, False)

                        # get the noise for the student and teacher
                        student_noise = torch.randn(
                            *constants.LATENT_SHAPE(self.bs),
                            generator=generator
                        ).to(constants.DEVICE)
                        if self.constant_noise:
                            teacher_noise = self.noise.expand(self.bs, -1, -1, -1)
                        else:
                            teacher_noise = student_noise

                        # get the model output
                        student_output = student_unet(
                            student_noise,
                            student_t_vec,
                            prompt_embeds.float(),
                        ).sample
                        x = get_original_sample(
                            pipe.scheduler,
                            student_output,
                            student_t_vec,
                            student_noise
                        )

                        # get the sample and teacher output 
                        with torch.no_grad():

                            # sample timestep and apply noise
                            t = torch.randint(
                                    0,
                                    num_t,
                                    [self.bs],
                                    generator=generator,
                                    dtype=torch.long
                            ).to(constants.DEVICE)
                            x_t = pipe.scheduler.add_noise(
                                x, teacher_noise, t
                            )

                            # get inputs for the teacher
                            embeds_in = torch.cat(
                                    [
                                            prompt_embeds,
                                            negative_embeds
                                    ],
                                    dim=0
                            )
                            x_t_in = torch.cat([x_t]*2, dim=0)
                            t_in = torch.cat([t]*2, dim=0)

                            # get the guided teacher noise prediction
                            teacher_pred = pipe.unet(
                                    x_t_in, t_in, embeds_in,
                            ).sample
                            cond_pred, neg_pred = teacher_pred.chunk(2, dim=0)

                            teacher_output = neg_pred + self.guidance_scale * (cond_pred - neg_pred)

                        # compute the loss
                        loss = boltflow_loss(
                            pipe.scheduler,
                            teacher_output,
                            t,
                            x_t,
                            x
                        )

                        loss = loss / self.grad_accum_steps
                        loss_accum += loss.item()

                    # backprop
                    if enable_autocast:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(True)

                # logging
                self.loss_log.append(loss_accum)
                pbar.set_postfix({"loss": loss_accum, "lr": lr_scheduler.get_last_lr()[0]})

                # save progress
                if (step+1) % self.eval_freq == 0 or step == (self.num_steps - 1) or step == 0:
                    self.evaluate(pipe, student_unet, step+1)
                    if not self.debug:
                        self.save_log()
                if (step+1) % self.checkpoint_freq == 0 or step == (self.num_steps - 1):
                    self.save_checkpoint(student_unet)
