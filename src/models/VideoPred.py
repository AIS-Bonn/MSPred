"""
Abstract class for video prediction models.
Directly suitable for models with the following computational graph:
   Input –> Encoder –> Predictor (+ Prior and Posterior) –> Decoder –> Output

More complex models (e.g. hierarchical and ladder networks) must override this forward pass
"""

import torch
import torch.nn as nn
from CONFIG import ENCODERS


class VideoPredModel(nn.Module):
    """
    Abstract class for video prediction models

    Args:
    -----
    model_params: dict
        Models section of the experiment parameters
    linear: bool
        Indicates whether the model will use linear or convolutional LSTMs
    """

    def __init__(self, model_params, linear=True, **kwargs):
        """ Model intializer """
        super().__init__()
        self.model_params = model_params
        self.linear = linear
        self.device_param = nn.Parameter(torch.empty(0))  # dummy param to fetch the device from
        batch_size = kwargs.get("batch_size", None)

        # modules
        self.encoder, self.decoder = self._get_encoder_decoder()
        self.predictor, self.prior, self.posterior = None, None, None
        self.hidden = self.init_hidden(batch_size) if batch_size is not None else None

        # hierarchical-specific parameters
        self.counters = None
        self.period = None
        return

    def forward(self, x, context=4, pred_frames=10, teacher_force=False):
        """
        Basic forward pass for a video prediction model. It is overridden in hierarch models

        Args:
        -----
        x: torch Tensor
            Batch of sequences to feed to the model. Shape is (B, Frames, C, H, W)
        context: integer
            number of seed frames to give as context to the model
        pred_frames: integer
            number of frames to predict. #frames=pred_frames are predicted autoregressively
        teacher_force: boolean
            If True, real frame is given as input during autoregressive prediction mode

        Returns:
        --------
        out_dict: dictionary
            dict containing predicted frames, and means and variances for
            the prior and posterior distributions
        """
        batch_size, num_frames, num_channels, in_H, in_W = x.shape
        self.init_hidden(batch_size=batch_size)
        self.init_counter()
        out_dict = {"preds": None, "target_masks": torch.full((num_frames,), False),
                    "mu_post": [], "logvar_post": [], "mu_prior": [], "logvar_prior": [],
                    "latents": []}

        inputs = x[:, :].float()
        targets = x[:, 1:].float()
        next_input = inputs[:, 0]  # first frame
        preds = []
        for t in range(0, context + pred_frames-1):
            # encoding image
            target_feats, _ = self.encoder(targets[:, t]) if (t < num_frames-1) else (None, None)
            if (t < context):
                feats, skips = self.encoder(next_input)
            else:
                feats, _ = self.encoder(next_input)
            # predicting latent and learning distribution
            if (self.prior is not None and self.posterior is not None):
                if target_feats is not None:
                    (latent_post, mu_post, logvar_post), _ = self.posterior(target_feats)
                else:
                    latent_post, mu_post, logvar_post = None, None, None
                (latent_prior, mu_prior, logvar_prior), _ = self.prior(feats)
                latent = latent_post if (t < context-1 or self.training) else latent_prior
                out_dict["mu_post"].append(mu_post)
                out_dict["logvar_post"].append(logvar_post)
                out_dict["mu_prior"].append(mu_prior)
                out_dict["logvar_prior"].append(logvar_prior)
                out_dict["latents"].append(latent)
                feats = torch.cat([feats, latent], 1)

            # predicting future features and decoding next frame
            pred_feats = self.predictor(feats)
            pred_output, _ = self.decoder([pred_feats, skips])
            if (t >= context-1):
                preds.append(pred_output)
                out_dict["target_masks"][t+1] = True

            # feeding GT in context or teacher-forced mode, autoregressive o/w
            next_input = inputs[:, t+1] if (t < context-1 or teacher_force) else pred_output

        preds = torch.stack(preds, dim=1)
        out_dict["preds"] = preds
        return out_dict

    def init_hidden(self, batch_size):
        """ Basic logic for initializing hidden states. It's overriden in hierarch models"""
        device = self.device_param.device
        img_size = self.model_params["img_size"]
        input_size = self.encoder.get_spatial_dims(img_size, -1)
        self.predictor.hidden = self.predictor.init_hidden(
                batch_size=batch_size,
                device=device,
                input_size=input_size
            )
        if(self.prior is not None):
            self.prior.hidden = self.prior.init_hidden(
                    batch_size=batch_size,
                    device=device,
                    input_size=input_size
                )
        if(self.posterior is not None):
            self.posterior.hidden = self.posterior.init_hidden(
                    batch_size=batch_size,
                    device=device,
                    input_size=input_size
                )
        return

    def _get_predictor(self):
        """ Module for instanciating the predictor model """
        raise NotImplementedError("Abstract class does not implement '_get_predictor'")

    def _get_prior_post(self):
        """ Module for instanciating the prior or posterior model """
        raise NotImplementedError("Abstract class does not implement '_get_prior_post'")

    def _get_encoder_decoder(self):
        """ Instanciating the encoder and decoder """
        num_channels = self.model_params["num_channels"]
        enc_type = self.model_params["enc_dec_type"]
        enc_dim = dec_dim = self.model_params["enc_dec"]["dim"]
        num_filters = self.model_params["enc_dec"]["num_filters"]
        extra_deep = self.model_params["enc_dec"]["extra_deep"]
        deeper_enc = self.linear and ("SpatioTempHierarch" not in self.model_params["model_type"])
        if enc_type == "DCGAN":
            if deeper_enc:
                import models.DCGAN_64 as enc_dec_models
            else:
                import models.DCGAN_Conv as enc_dec_models
            encoder = enc_dec_models.encoder(dim=enc_dim, nf=num_filters, nc=num_channels)
            decoder = enc_dec_models.decoder(dim=dec_dim, nf=num_filters, nc=num_channels)
        elif enc_type == "VGG":
            if deeper_enc:
                import models.VGG_64 as enc_dec_models
            else:
                import models.VGG_Conv as enc_dec_models
            encoder = enc_dec_models.encoder(dim=enc_dim, nc=num_channels, extra_deep=extra_deep)
            decoder = enc_dec_models.decoder(dim=dec_dim, nc=num_channels, extra_deep=extra_deep)
        else:
            raise NotImplementedError(f"Unknow model {enc_type} not in {ENCODERS}")
        return encoder, decoder

    def check_counters(self):
        """ Getting the state of the counters and periods """
        raise NotImplementedError("'check_counters()' method is not available in the abstract class")

    def init_counter(self):
        """ Initializing counters with zeros. This is needed at the beginning of each iteration"""
        self._init_module(self.predictor)
        if(self.prior is not None):
            self._init_module(self.prior)
        if(self.posterior is not None):
            self._init_module(self.posterior)
        return

    def reset_counter(self):
        """ Resetting period counter """
        self._reset_module(self.predictor)
        if(self.prior is not None):
            self._reset_module(self.prior)
        if(self.posterior is not None):
            self._reset_module(self.posterior)
        return

    def _init_module(self, module):
        """ Re-initializing period counter with zeros for one particular module or module list"""
        if isinstance(module, nn.ModuleList):
            for m in module:
                m.init_counter()
        elif isinstance(module, nn.Module):
            module.init_counter()
        else:
            raise ValueError("")

    def _reset_module(self, module):
        """ Resetting period counter for one particular module or module list"""
        if isinstance(module, nn.ModuleList):
            for m in module:
                m.reset_counter()
        elif isinstance(module, nn.Module):
            module.reset_counter()
        else:
            raise ValueError("")


#
