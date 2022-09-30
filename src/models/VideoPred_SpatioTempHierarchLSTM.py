"""
Spatio-Temporal Hierarchical Multi-Scale Stochastic Video Prediction model.
This corresponds to our proposed MSPred model
"""

import torch
import torch.nn as nn
import models


class SpatioTempHierarch(models.HierarchModel):
    """ MSPred model """

    def __init__(self, model_params, linear=True, stochastic=False, use_output=True, **kwargs):
        """ MSPred model intializer """
        super().__init__(model_params=model_params, linear=linear,
                         stochastic=stochastic, use_output=use_output, **kwargs)
        assert self.num_hierarch == 3, "Invalid number of hierarchy levels!"
        assert not (self.last_context_residuals and self.linear), "Residual-connections used with linear-LSTMs!"
        assert not (self.ancestral_sampling and self.linear), "Ancestral-sampling used with linear-LSTMs!"
        return

    def predict(self, feats_dict, out_dict, cur_frame, context):
        """
        Predicting next features using current ones. Corresponds to the forward
        pass through our hierarchical predictor module.

        Args:
        -----
        feats_dict: dict
            Dict. containing tensor with the input and target features for each level in the hiearchy
        out_dict: dict
            Dict containing lists where all outputs and intermediate values are stored
        cur_frame: int
            Index of the current frame in the sequence
        context: int
            Number of context frames to use
        """
        feats = feats_dict["input"]
        target_feats = feats_dict["target"]

        # sampling latent vectors for each level in the hierarchy using Gaussian LSTMs
        if (self.stochastic):
            for h in reversed(range(self.num_hierarch)):
                post_input = target_feats[h]
                prior_input = feats[h]
                # Ancestral sampling: each level latent also conditioned on all upper level samples.
                if self.ancestral_sampling and h != self.num_hierarch-1:
                    prev_latents = [latent_list[-1] for latent_list in out_dict["latents"][h+1:]]
                    # Spatially upsample 2x, 4x, ... and concatenate latent samples from upper levels.
                    for i, z in enumerate(prev_latents):
                        prev_latents[i] = nn.functional.interpolate(z, scale_factor=2**(i+1))
                    prev_latents = torch.cat(prev_latents, dim=1)
                    post_input = torch.cat([post_input, prev_latents], dim=1)
                    prior_input = torch.cat([prior_input, prev_latents], dim=1)

                # forward through prior and posterion gaussian LSTMs
                (latent_post, mu_post, logvar_post), ticked = self.posterior[h](post_input)
                if not ticked:
                    continue
                (latent_prior, mu_prior, logvar_prior), _ = self.prior[h](prior_input)
                latent = latent_post if (cur_frame < context-1 or self.training) else latent_prior
                out_dict["latents"][h].append(latent)
                if cur_frame >= context-1:
                    out_dict["mu_post"][h].append(mu_post)
                    out_dict["logvar_post"][h].append(logvar_post)
                    out_dict["mu_prior"][h].append(mu_prior)
                    out_dict["logvar_prior"][h].append(logvar_prior)

        assert len(feats) >= len(self.predictor)
        # predicting next features for each level in the hierarchy
        pred_outputs = []
        for h, cur_model in enumerate(self.predictor):
            feats_ = feats[h]
            if self.stochastic:
                latent = out_dict["latents"][h][-1]
                if self.linear:
                    feats_ = torch.cat([feats_.reshape(-1, cur_model.input_size-latent.shape[1]), latent], 1)
                else:
                    feats_ = torch.cat([feats_, latent], 1)
            pred_feats = cur_model(feats_, hidden_state=cur_model.hidden[0])
            pred_outputs.append(pred_feats)
        if self.linear:
            pred_outputs = self._reshape_preds(pred_outputs)

        return pred_outputs, out_dict

    def _reshape_preds(self, preds):
        """ Reshaping predicted feature vectors before passing to decoder """
        assert self.linear is True
        assert len(preds) == self.num_hierarch
        img_size = self.model_params["img_size"]
        nf_enc = self.model_params["enc_dec"]["num_filters"]
        for n in range(self.num_hierarch):
            C = nf_enc * 2**(n+1)
            H, W = self.encoder.get_spatial_dims(img_size, n+1)
            preds[n] = torch.reshape(preds[n], (-1, C, H, W))
        return preds

    def _get_input_feats(self, enc_outs, enc_skips):
        return [*enc_skips[1:], enc_outs]

    def _get_residual_feats(self, enc_outs, enc_skips):
        return [*enc_skips, enc_outs]

    def _get_decoder_inputs(self, pred_feats, residuals):
        dec_input_feats = [residuals[0]]
        if not self.last_context_residuals:
            dec_input_feats = dec_input_feats + pred_feats
        else:
            for i, feat in enumerate(pred_feats):
                dec_input_feats.append(torch.add(feat, residuals[i+1]))
        return [dec_input_feats[-1], dec_input_feats[:-1]]

    def _get_decoder_head_inputs(self, pred_feats, dec_skips):
        return dec_skips[-2::-1]

    def _get_predictor(self):
        """ Instanciating the temporal-hierarchy prediction model """
        pred_model = models.LSTM if self.linear else models.ConvLSTM
        nf_enc = self.model_params["enc_dec"]["num_filters"]
        img_size = self.model_params["img_size"]

        predictor = []
        for n in range(self.num_hierarch):
            input_size = output_size = nf_enc * 2**(n+1)
            if self.linear:
                h, w = self.encoder.get_spatial_dims(img_size, n+1)
                input_size = input_size * (h * w)
            output_size = input_size
            if self.stochastic:
                input_size += self.model_params["LSTM_Prior"]["latent_dim"]
            predictor.append(
                pred_model(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_lstm_layers[n],
                    period=self.periods[n],
                    use_output=self.use_output
                )
            )
        predictor = nn.ModuleList(predictor)
        return predictor

    def _get_prior_post(self, model_key):
        """
        Instanciating the modules to estimate the posterior and prior distributions.
        We use different recurrent models for each level of the temporal hierarchy.
        """
        assert model_key in ["LSTM_Prior", "LSTM_Posterior"]
        prior_model = models.Gaussian_LSTM if self.linear else models.GaussianConvLSTM
        nf_enc = self.model_params["enc_dec"]["num_filters"]
        z_dim = self.model_params["LSTM_Prior"]["latent_dim"]
        hid_dim = self.model_params[model_key]["hidden_dim"]
        n_layers = self.model_params[model_key]["num_layers"]
        img_size = self.model_params["img_size"]
        modules = []
        for n in range(self.num_hierarch):
            input_size = nf_enc * 2**(n+1)
            if self.linear:
                h, w = self.encoder.get_spatial_dims(img_size, n+1)
                input_size = input_size * (h * w)
            if self.ancestral_sampling:
                input_size += (z_dim * (self.num_hierarch-n-1))
            modules.append(
                prior_model(
                    input_size=input_size,
                    output_size=z_dim,
                    hidden_size=hid_dim,
                    num_layers=n_layers,
                    period=self.periods[n]
                )
            )
        model = nn.ModuleList(modules)
        return model

    def _get_decoder_heads(self):
        """ Instanciating decoder heads for predicting high-level representations """
        nf = self.model_params["enc_dec"]["num_filters"]
        out_channels = self.n_hmap_channels
        decoder_heads = nn.ModuleList()
        for h in range(1, self.num_hierarch):
            decoder_heads.append(
                    models.DeconvHead(
                            in_channels=nf*(2**h),
                            num_filters=nf,
                            out_channels=out_channels[h-1],
                            num_layers=h+1,
                            period=self.periods[h]
                        )
                )
        return decoder_heads

    def init_hidden(self, batch_size):
        """ Initializing hidden states for all recurrent models """
        device = self.device_param.device
        img_size = self.model_params["img_size"]
        for h, m in enumerate(self.predictor):
            _ = m.init_hidden(
                    batch_size=batch_size,
                    device=device,
                    input_size=self.encoder.get_spatial_dims(img_size, h+1)
                )
        if self.stochastic:
            for h, m in enumerate(self.prior):
                _ = m.init_hidden(
                        batch_size=batch_size,
                        device=device,
                        input_size=self.encoder.get_spatial_dims(img_size, h+1)
                    )
            for h, m in enumerate(self.posterior):
                _ = m.init_hidden(
                        batch_size=batch_size,
                        device=device,
                        input_size=self.encoder.get_spatial_dims(img_size, h+1)
                    )
        return

#
