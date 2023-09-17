"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import random
from typing import Optional, Union, List, Callable, Dict, Any

import os
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import cm.dist_util as dist
import random


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return th.cat([x, x.new_zeros([1])])


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_weightings(weight_schedule, snrs, sigma_data=0.5):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


def get_snr(sigma):
    return (1 - sigma ** 2) / sigma ** 2


def finalize_consistency_losses(
    distiller,
    distiller_target,
    sigma: torch.Tensor,
    loss_norm: str = 'l2',
    weight_schedule: str = 'karras',
    sigma_data: float = 0.5
):
    snrs = get_snr(sigma)
    weights = get_weightings(weight_schedule, snrs, sigma_data)
    if loss_norm == "l1":
        diffs = th.abs(distiller - distiller_target)
        loss = mean_flat(diffs) * weights
    elif loss_norm == "l2":
        diffs = (distiller - distiller_target) ** 2
        loss = mean_flat(diffs) * weights
    elif loss_norm == "l2-32":
        distiller = F.interpolate(distiller, size=32, mode="bilinear")
        distiller_target = F.interpolate(
            distiller_target,
            size=32,
            mode="bilinear",
        )
        diffs = (distiller - distiller_target) ** 2
        loss = mean_flat(diffs) * weights
    else:
        raise ValueError(f"Unknown loss norm {loss_norm}")
    
    terms = {}
    terms["loss"] = loss
    return terms


class DenoiserSD:
    def __init__(
        self,
        pipe,
        sigma_data: float = 0.5,
        weight_schedule="uniform",
        loss_norm="l2",
        num_timesteps=50,
        use_fp16=False
    ):
        self.pipe = pipe
        self.scheduler = pipe.scheduler
        self.sigmas = (1 - self.scheduler.alphas_cumprod) ** 0.5
        self.sigma_data = sigma_data
        self.sigma_max = max(self.sigmas)
        self.sigma_min = min(self.sigmas)
        self.weight_schedule = weight_schedule
        self.loss_norm = loss_norm
        self.num_timesteps = num_timesteps
        self.use_fp16 = use_fp16

        self.generator = torch.Generator(
            device=self.pipe._execution_device
        ).manual_seed(dist.get_seed())

    def consistency_losses(
        self,
        ddp_model,
        teacher_model, 
        target_model, 
        image: Union[
            torch.FloatTensor,
            np.ndarray,
            List[torch.FloatTensor],
            List[np.ndarray],
        ], 
        prompt: Union[str, List[str]],
        guidance_scale: float = 8.0,
        latents: Optional[torch.FloatTensor] = None,
        num_scales: int = 50,
        **kwargs
    ):
        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(prompt, 0, 1, None, None, None)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        # replace 10% of text prompts with "" for classifier-free guidance
        # filtered_prompt = []
        # for p in prompt:
        #     if random.random() > cfg_prob:
        #         filtered_prompt.append(p)
        #     else:
        #         filtered_prompt.append("")

        device = self.pipe._execution_device
        
        # 3. Encode input prompt
        # with torch.no_grad():
        #     filtered_prompt_embeds = self.pipe._encode_prompt(
        #         filtered_prompt,
        #         device,
        #         1,
        #         False # do_classifier_free_guidance
        #     )
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        with torch.no_grad():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device,
                1, do_classifier_free_guidance
            )

        # 4. Sample timesteps uniformly
        self.scheduler.set_timesteps(num_scales, device=device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device) 
        indices = th.randint(
            0, num_scales - 1, (batch_size,), generator=self.generator, device=device
        )
        t  = self.scheduler.timesteps[indices]
        t2 = self.scheduler.timesteps[indices + 1]
        
        # 5. Prepare latent variables
        with torch.no_grad():
            latents = self.pipe.prepare_latents(
                image, t, batch_size, 1, prompt_embeds.dtype, device, self.generator
            )
    
        # 6 Get x_0(x_t) using the distilled model
        assert ddp_model.module.training
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        distiller_noise_pred = ddp_model(
            latent_model_input, #latents,
            torch.cat([t] * 2) if do_classifier_free_guidance else t, #t,
            encoder_hidden_states=prompt_embeds,#filtered_prompt_embeds,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            distiller_noise_pred_uncond, distiller_noise_pred_text = distiller_noise_pred.chunk(2)
            distiller_noise_pred = distiller_noise_pred_uncond + guidance_scale * (distiller_noise_pred_text - distiller_noise_pred_uncond)

        distiller = self.denoise(distiller_noise_pred, t, latents)

        # 7 Get x_t-1 using the teacher model
        with torch.no_grad():
            teacher_noise_pred = teacher_model(
                latent_model_input.to(torch.float16),
                torch.cat([t] * 2) if do_classifier_free_guidance else t,
                encoder_hidden_states=prompt_embeds.to(torch.float16),
                return_dict=False,
            )[0].to(torch.float32)
            
            # perform guidance
            if do_classifier_free_guidance:
                teacher_noise_pred_uncond, teacher_noise_pred_text = teacher_noise_pred.chunk(2)
                teacher_noise_pred = teacher_noise_pred_uncond + guidance_scale * (teacher_noise_pred_text - teacher_noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents_prev = self.scheduler_step(teacher_noise_pred, t, t2, latents)
            if self.use_fp16:
                latents_prev = latents_prev.half()
            
        # 8 Get x_0(x_t-1) using target distilled model
        with torch.no_grad():
            latent_prev_model_input = torch.cat([latents_prev] * 2) if do_classifier_free_guidance else latents_prev
            distiller_target_noise_pred = target_model(
                latent_prev_model_input, #latents_prev,
                torch.cat([t2] * 2) if do_classifier_free_guidance else t2,
                encoder_hidden_states=prompt_embeds,#filtered_prompt_embeds,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                distiller_target_noise_pred_uncond, distiller_target_noise_pred_text = distiller_target_noise_pred.chunk(2)
                distiller_target_noise_pred = distiller_target_noise_pred_uncond + \
                    guidance_scale * (distiller_target_noise_pred_text - distiller_target_noise_pred_uncond)

            distiller_target = self.denoise(distiller_target_noise_pred, t2, latents_prev)

        sigma = (1 - self.scheduler.alphas_cumprod[t]) ** 0.5
        return finalize_consistency_losses(
            distiller, distiller_target, sigma,
            weight_schedule=self.weight_schedule, 
            **kwargs
        ), indices
    
    def denoise(self, epsilon, timestep, sample):
        assert self.scheduler.config.prediction_type == "epsilon"
        # 1. compute alphas, betas
        dims = sample.ndim
        alpha_prod_t = torch.where(
            timestep > 1, 
            self.scheduler.alphas_cumprod[timestep], 
            self.scheduler.final_alpha_cumprod
        ) 
        alpha_prod_t = append_dims(alpha_prod_t, dims)
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        denoised_sample = (sample - beta_prod_t ** (0.5) * epsilon) / alpha_prod_t ** (0.5)
        return denoised_sample

    def scheduler_step(
        self, 
        pred_epsilon: torch.FloatTensor,
        timestep: torch.IntTensor, 
        prev_timestep: torch.IntTensor,
        sample: torch.FloatTensor,
    ):
        assert self.scheduler.config.prediction_type == "epsilon"
        assert (prev_timestep >= 0).all() and (timestep > 0).all()
        dims = sample.ndim

        # 1. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t = append_dims(alpha_prod_t, dims)
        alpha_prod_t_prev = torch.where(
            prev_timestep > 0, self.scheduler.alphas_cumprod[prev_timestep], self.scheduler.final_alpha_cumprod
        ) 
        alpha_prod_t_prev  = append_dims(alpha_prod_t_prev, dims)
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon) / alpha_prod_t ** (0.5)

        # 3. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 4. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return prev_sample

    @torch.no_grad()
    def sample_with_my_step(self, 
        eval_pipe,
        prompt, 
        generator=None, 
        num_inference_steps=50, 
        guidance_scale=8.0,
        num_scales=50,
    ):
        height = eval_pipe.unet.config.sample_size * eval_pipe.vae_scale_factor
        width = eval_pipe.unet.config.sample_size * eval_pipe.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        eval_pipe.check_inputs(
            prompt, height, width, 1, None, None, None
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        device = eval_pipe._execution_device
        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = eval_pipe._encode_prompt(
            prompt,
            device,
            1,
            do_classifier_free_guidance
        )

        # 4. Sample timesteps uniformly (first step is 981)
        self.scheduler.set_timesteps(num_scales, device=device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device) 
        if num_inference_steps == num_scales:
            timesteps = self.scheduler.timesteps
        else:
            step = num_scales / num_inference_steps
            step_ids = torch.arange(0, num_scales, step).to(int)
            timesteps = self.scheduler.timesteps[step_ids]
        assert len(timesteps) == num_inference_steps

        # 5. Prepare latent variables
        num_channels_latents = eval_pipe.unet.config.in_channels
        latents = eval_pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        with eval_pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = eval_pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = eval_pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if i == len(timesteps) - 1:
                    t2 = torch.zeros_like(timesteps[i])
                else:
                    t2 = timesteps[i + 1]
                latents = self.scheduler_step(noise_pred, t, t2, latents)
                # call the callback, if provided
                progress_bar.update()

        image = eval_pipe.vae.decode(latents / eval_pipe.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = eval_pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        return image

    @torch.no_grad()
    def stochastic_iterative_sampler(
        self,
        eval_pipe,
        prompt,
        generator,
        ts=None,
        timesteps=None,
        guidance_scale=8.0,
        num_scales=50,
        num_inference_steps=3,
        with_refining=False,
    ):
        height = eval_pipe.unet.config.sample_size * eval_pipe.vae_scale_factor
        width = eval_pipe.unet.config.sample_size * eval_pipe.vae_scale_factor
        torch.manual_seed(dist.get_seed())

        # 1. Check inputs. Raise error if not correct
        eval_pipe.check_inputs(
            prompt, height, width, 1, None, None, None
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        device = eval_pipe._execution_device
        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = eval_pipe._encode_prompt(
            prompt,
            device,
            1,
            do_classifier_free_guidance
        )

        # 4. Sample timesteps uniformly (first step is 981)
        self.scheduler.set_timesteps(num_scales, device=device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

        if timesteps is None:
            if num_inference_steps == num_scales:
                timesteps = self.scheduler.timesteps
            elif ts is None:
                step = num_scales / num_inference_steps
                step_ids = torch.arange(0, num_scales, step).to(int)
                timesteps = self.scheduler.timesteps[step_ids]
                timesteps = torch.cat([timesteps, self.scheduler.timesteps[-1:]])
            else:
                timesteps = self.scheduler.timesteps[ts]

        assert len(timesteps) == num_inference_steps + 1

        # 5. Prepare latent variables
        num_channels_latents = eval_pipe.unet.config.in_channels
        latents = eval_pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        with eval_pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i == len(timesteps) - 1:
                    break
                next_t = timesteps[i + 1]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = eval_pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = eval_pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                x0 = self.denoise(noise_pred, t, latents)

                sqrt_alpha_prod = self.scheduler.alphas_cumprod[next_t] ** 0.5
                sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[next_t]) ** 0.5

                noise = torch.randn(latents.size(), device=device)
                latents = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

                # call the callback, if provided
                progress_bar.update()
        if not with_refining:
            image = eval_pipe.vae.decode(latents / eval_pipe.vae.config.scaling_factor, return_dict=False)[0]
            do_denormalize = [True] * image.shape[0]
            image = eval_pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        else:
            image = None
        return image, latents, prompt_embeds

    @torch.no_grad()
    def refining(self,
                 eval_pipe,
                 prompt_embeds,
                 x0_latents,
                 rollback_value=0.3,
                 generator=None,
                 num_inference_steps=50,
                 guidance_scale=8.0,
                 ):
        device = eval_pipe._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Sample timesteps uniformly (first step is 981)
        timesteps = torch.linspace(int(rollback_value * 1000), 1, steps=num_inference_steps + 1,
                                   dtype=int, device=device)
        assert len(timesteps) == num_inference_steps + 1

        # 5. Prepare latent variables
        rollback_timestep = torch.tensor([int(rollback_value * 1000)], device=device)
        sqrt_alpha_prod = self.scheduler.alphas_cumprod[rollback_timestep] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[rollback_timestep]) ** 0.5

        noise = torch.randn(x0_latents.size(), generator=generator, device=device)
        latents = sqrt_alpha_prod * x0_latents + sqrt_one_minus_alpha_prod * noise

        with eval_pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = eval_pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = eval_pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if i == len(timesteps) - 1:
                    t2 = torch.zeros_like(timesteps[i])
                else:
                    t2 = timesteps[i + 1]
                latents = self.scheduler_step(noise_pred, t, t2, latents)
                # call the callback, if provided
                progress_bar.update()

        image = eval_pipe.vae.decode(latents / eval_pipe.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = eval_pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        return image
