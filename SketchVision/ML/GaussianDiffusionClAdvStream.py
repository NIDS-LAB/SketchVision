# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
import random

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from helpers import *
import torch.nn.functional as F
import torchvision.transforms as T


def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas


def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL Divergence between two gaussians

    :param mean1:
    :param logvar1:
    :param mean2:
    :param logvar2:
    :return: KL Divergence between N(mean1,logvar1^2) & N(mean2,logvar2^2))
    """
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretised_gaussian_log_likelihood(x, means, log_scales):
    """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
            )
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusionModel:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",  # l2,l1 hybrid
            loss_weight='none',  # prop t / uniform / None
            noise="gauss",
            timestep_map=None,
            sel_attn_depth=8,
            sel_attn_block="output",
            num_heads=4,
            ):
        super().__init__()
        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)

        else:
            raise NotImplementedError(f"unknown noise type: {noise}")

        self.img_size = img_size
        self.sel_attn_depth = sel_attn_depth
        self.sel_attn_block = sel_attn_block
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_heads = num_heads
        self.num_timesteps = len(betas)
        if timestep_map is None:
            self.map_tensor = torch.tensor(np.array(list(range(len(betas)))))
        else:
            self.map_tensor = torch.tensor(timestep_map)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)

        self.loss_weight = loss_weight
        alphas = 1 - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
                np.append(self.posterior_variance[1], self.posterior_variance[1:])
                )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

        height = width = img_size[0]
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2
        sigma = 0.05
        weights = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * (sigma * height) ** 2))
        self.guassian_weights = weights / weights.max() 
        self.guassian_weights = (self.guassian_weights > 0.1).astype(int)


    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
               / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def predict_x_0_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
                self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
                )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, model, x_t, t, label, estimate_x_0=None, attn_map=None, pnum=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))

        :param model:
        :param x_t:
        :param t:
        :return:
        """
        self.map_tensor = self.map_tensor.to(x_t.device)
        if estimate_x_0 == None:
            estimate_x_0, attn_map = model(x_t, self.map_tensor[t], label, pnum)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        #pred_x_0 = self.predict_x_0_from_xprev(x_t, t, estimate_x_t_minus_1).clamp(-1, 1)
        mean, _, _ = self.q_posterior_mean_variance(estimate_x_0, x_t, t)
        
        return {
            "mean":         mean,
            "variance":     model_var,
            "log_variance": model_logvar,
            "pred_x_0":     estimate_x_0.clamp(-1, 1),
            "eps":  None,
            "attn_map" : attn_map,
            }

    def attention_masking(
        self, x, t, attn_map, prev_noise, blur_sigma,):
        """
        Apply the self-attention mask to produce bar{x_t}

        :param x: the predicted x_0 [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param attn_map: the attention map tensor at time t.
        :param prev_noise: the previously predicted epsilon to inject
            the same noise as x_t.
        :param blur_sigma: a sigma of Gaussian blur.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: the bar{x_t}
        """
        B, C, H, W = x.shape
        assert t.shape == (B,)
        
        if self.sel_attn_depth in [0, 1, 2] or self.sel_attn_block == "middle":
            attn_res = 8
        elif self.sel_attn_depth in [3, 4, 5]:
            attn_res = 16
        elif self.sel_attn_depth in [6, 7, 8]:
            attn_res = 32
        else:
            raise ValueError("sel_attn_depth must be in [0, 1, 2, 3, 4, 5, 6, 7, 8]")

        # Generating attention mask
        attn_mask = attn_map.reshape(B, self.num_heads, attn_res ** 2, attn_res ** 2).mean(1, keepdim=False).sum(1, keepdim=False) > 0.9
        attn_mask = attn_mask.reshape(B, attn_res, attn_res).unsqueeze(1).repeat(1, 3, 1, 1).int().float()
        attn_mask = F.interpolate(attn_mask, (H, W))

        # Gaussian blur
        transform = T.GaussianBlur(kernel_size=31, sigma=blur_sigma)
        x_curr = transform(x)

        # Apply attention masking
        x_curr = x_curr *(attn_mask) + x * (1 - attn_mask)
        #x_curr =  x_curr * (w)

        # Re-inject the noise
        x_curr = self.sample_q(x_curr, t, noise=prev_noise)
        
        return x_curr, attn_mask

    
    def sample_p(self, model, x_t, t, denoise_fn="gauss", label=None, pnum=None, guidance_kwargs=None):


        if guidance_kwargs is not None:
            guide_scale = guidance_kwargs['guide_scale']
            guide_start = guidance_kwargs['guide_start']
            blur_sigma = guidance_kwargs['blur_sigma']
            
            out = self.p_mean_variance(model, x_t, t, label)
            cond_eps = out['eps']

            mask_blurred, attn_mask = self.attention_masking(out['pred_x_0'], t, out['attn_map'], prev_noise=cond_eps, blur_sigma=blur_sigma,)
            mask_out = self.p_mean_variance(model, mask_blurred, t, label)
            uncond_eps = mask_out['eps']
            guided_eps = uncond_eps + guide_scale * (cond_eps - uncond_eps)

            pred_xstart = self.predict_x_0_from_eps(x_t, t, guided_eps).clamp(-1, 1)

            final_out = {}
            final_out["mean"], _, _ = self.q_posterior_mean_variance(pred_xstart, x_t, t)
            final_out["variance"] = out["variance"]
           
            noise = torch.randn_like(x_t)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
            )  # no noise when t == 0
            sample = final_out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            return {"sample": sample, "pred_x_0": out["pred_x_0"], "attn_mask": attn_mask}

        else:
            out = self.p_mean_variance(model, x_t, t, label, None, None, pnum)
            # noise = torch.randn_like(x_t)
            if type(denoise_fn) == str:
                if denoise_fn == "gauss":
                    noise = torch.randn_like(x_t)
                elif denoise_fn == "noise_fn":
                    noise = self.noise_fn(x_t, t).float()
                elif denoise_fn == "random":
                    noise = torch.randn_like(x_t)
            else:
                noise = denoise_fn(x_t, t)
    
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
            )
            sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            return {"sample": sample, "pred_x_0": out["pred_x_0"]}
   
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t - pred_xstart) \
        / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)


    
    def ddim_sample(self, model, x_t, t, denoised_fn=None, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(model, x_t, t, None)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x_t, t, out["pred_x_0"])

        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape, x_t.device)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x_t.shape, x_t.device)
        sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        noise = th.randn_like(x_t)
        mean_pred = (out["pred_x_0"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def forward_backward(
            self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss", label=None, pnum=None, guidance_kwargs=None, ddim=False,
            ):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        seq_atn = []
        if see_whole_sequence == "whole":

            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                # noise = torch.randn_like(x)
                noise = self.noise_fn(x, t_batch).float()
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())
        else:
            # x = self.sample_q(x,torch.tensor([t_distance], device=x.device).repeat(x.shape[0]),torch.randn_like(x))
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
            x = self.sample_q(
                    x, t_tensor,
                    self.noise_fn(x, t_tensor).float()
                    )
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())
    
        for t in range(int(t_distance) -1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                if ddim:
                    #ts = map_tensor[t].cpu().item()
                    out = self.ddim_sample(model, x, t_batch, denoise_fn)        
                else:
                    out = self.sample_p(model, x, t_batch, denoise_fn, label, pnum, guidance_kwargs)
                    if guidance_kwargs:
                        y = out["attn_mask"]
                        seq_atn.append(y.cpu().detach())
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        return x.detach() if not see_whole_sequence else seq, seq_atn

    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )

            :param x_0:
            :param t:
            :param noise:
            :return:
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def sample_q_gradual(self, x_t, t, noise):
        """
        q (x_t | x_{t-1})
        :param x_t:
        :param t:
        :param noise:
        :return:
        """
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, xc_t, x_t, t, estimate_x_0=None, label=None, pnum=None):
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, xc_t, t)
        output = self.p_mean_variance(model, x_t, t, label, estimate_x_0, None, pnum)
        kl = normal_kl(true_mean, true_log_var, output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretised_gaussian_log_likelihood(
                x_0, output["mean"], log_scales=0.5 * output["log_variance"]
                )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}

    def calc_loss(self, model, x_0, xa_0, t, label, pnum):
        # noise = torch.randn_like(x)

        noise = self.noise_fn(x_0, t).float()

        x_t = self.sample_q(xa_0, t, noise)
        xc_t = self.sample_q(x_0, t, noise)
        #estimate_x_t_minus_1, _ = model(x_t, t, label)
        #x_t_minus_1_clean = self.sample_q(x0, t - 1, noise)
        estimate_x_0, _ = model(x_t, t, label, pnum)

        loss = {}
        if self.loss_type == "l1":
            loss["loss"] = mean_flat((estimate_x_0 - x_0).abs())
        elif self.loss_type == "l2":
            loss["loss"] = mean_flat((estimate_x_0 - x_0).square())
        elif self.loss_type == "hybrid":
            # add vlb term
            loss["vlb"] = self.calc_vlb_xt(model, x_0, xc_t, x_t, t, estimate_x_0, label)["output"]
            loss["loss"] = loss["vlb"] + mean_flat((estimate_x_0 - x_0).square())
        else:
            loss["loss"] = mean_flat((estimate_x_0 - x_0).square())
        return loss, x_0, estimate_x_0

    def p_loss(self, model, x_0, xa_0, args, label=None, pnum=None):
        if self.loss_weight == "none":
            if args["train_start"]:
                t = torch.randint(
                        0, min(args["sample_distance"], self.num_timesteps), (x_0.shape[0],),
                        device=x_0.device
                        )
            else:
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            weights = 1
        else:
            t, weights = self.sample_t_with_weights(x_0.shape[0], x_0.device)

        loss, x_tm1, eps_t = self.calc_loss(model, x_0, xa_0, t, label, pnum)
        loss = ((loss["loss"] * weights).mean(), (loss, x_tm1, eps_t))
        return loss

    def prior_vlb(self, x_0, args):
        t = torch.tensor([self.num_timesteps - 1] * args["Batch_Size"], device=x_0.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
        kl_prior = normal_kl(
                mean1=qt_mean, logvar1=qt_log_variance, mean2=torch.tensor(0.0, device=x_0.device),
                logvar2=torch.tensor(0.0, device=x_0.device)
                )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_total_vlb(self, x_0, xa_0, model, args, label, pnum):
        vb = []
        x_0_mse = []
        mse = []
        for t in reversed(list(range(self.num_timesteps))):
            t_batch = torch.tensor([t] * args["Batch_Size"], device=x_0.device)
            noise = torch.randn_like(x_0)
            xc_t = self.sample_q(x_0=x_0, t=t_batch, noise=noise)
            x_t = self.sample_q(x_0=xa_0, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self.calc_vlb_xt(model, x_0, xc_t, x_t, t_batch, None, label, pnum)
                #self.calc_vlb_xt(model, x_0, xc_t, x_t, t, estimate_x_0, label)["output"]
            vb.append(out["output"])
            x_0_mse.append(mean_flat((out["pred_x_0"] - x_0) ** 2))
            eps = self.predict_eps_from_x_0(x_t, t_batch, out["pred_x_0"])
            mse.append(mean_flat((eps - noise) ** 2)) ### TODO: this is based on noise prediction, need to be changed

        vb = torch.stack(vb, dim=1)
        x_0_mse = torch.stack(x_0_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_vlb = self.prior_vlb(x_0, args)
        total_vlb = vb.sum(dim=1) + prior_vlb
        return {
            "total_vlb": total_vlb,
            "prior_vlb": prior_vlb,
            "vb":        vb,
            "x_0_mse":   x_0_mse,
            "mse":       mse,
            }

