import torch
import torch.nn.functional as F


@torch.no_grad()
def decode_image(
    pipe,
    latents: torch.FloatTensor
):
    """ Decode the latents into an image using the given pipe.

    Args:
        pipe (StableDiffusionPipeline): Pipe object containing the vae and image processor.
        latents (torch.FloatTensor): Latents to decode.

    Returns:
        np.ndarray: Decoded image.
    """

    # vae must be full precision
    og_dtype = pipe.vae.dtype
    if og_dtype != torch.float32:
        pipe.vae.to(torch.float32)

    # apply the vae and postprocess
    img = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor, return_dict=False
    )[0]
    img = pipe.image_processor.postprocess(img, output_type="np")

    # return vae to original dtype
    if og_dtype != torch.float32:
        pipe.vae.to(og_dtype)

    return img


def get_original_sample(
    scheduler,
    model_output: torch.FloatTensor,
    timestep: torch.LongTensor,
    sample: torch.FloatTensor,
):
    """ Get the original sample predicted by the model output and timestep.

    Args:
        scheduler (DDIMScheduler): Scheduler object used by the unet.
        model_output (torch.FloatTensor): Output of the unet.
        timestep (torch.LongTensor): Timestep vector given to the unet.
        sample (torch.FloatTensor): noised x_t given to the unet.
    """

    # error checking
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # 2. compute alphas and betas, make broadcastable
    alpha_prod_t = scheduler.alphas_cumprod.to(timestep.device)[timestep]
    while len(alpha_prod_t.shape) < len(sample.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )
    
    return pred_original_sample


def sds_loss():
    pass


def boltflow_loss(
    scheduler,
    model_output: torch.FloatTensor,
    timestep: torch.LongTensor,
    sample: torch.FloatTensor,
    x: torch.FloatTensor,
):
    """ BoltFlow loss function.
    Minimizes the logprob cost of the straight sde path from noise to x.
    Timesteps are weighted by beta_prod_t**(-1/2) as empirically found to work much better.

    Largely copied from DDIMScheduler.step()
    
    Args:
        scheduler (DDIMScheduler): Scheduler object used by the teacher unet.
        model_output (torch.FloatTensor): Output of the teacher unet.
        timestep (torch.LongTensor): Timestep vector given to the teacher unet.
        sample (torch.FloatTensor): noised x_t given to the teacher unet.
        x (torch.FloatTensor): x_0 that we are optimizing. 

    Returns:
        torch.FloatTensor: Loss value.
    """

    # error checking
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # compute alphas and betas, make broadcastable
    alpha_prod_t = scheduler.alphas_cumprod.to(timestep.device)[timestep]
    while len(alpha_prod_t.shape) < len(sample.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    beta_prod_t = 1 - alpha_prod_t

    # get the predicted original sample
    pred_original_sample = get_original_sample(scheduler, model_output, timestep, sample)

    # boltflow loss
    loss =  ((x - pred_original_sample)**2) / torch.sqrt(beta_prod_t)
    return loss.mean()
