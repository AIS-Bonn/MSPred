"""
Deterministic and Learned-Prior Models from "Stochastic Video Prediction with a Learned Prior".
Can be used with either linear or convolutional prediction models (LSTMs)
"""

import models


class SVG_DET(models.VideoPredModel):
    """ Deterministic SVG model intializer """

    def __init__(self, model_params, linear=True, **kwargs):
        """ Model intializer """
        super().__init__(model_params=model_params, linear=linear, **kwargs)
        self.predictor = self._get_predictor()
        return

    def forward(self, x, *args, **kwargs):
        """ Forward pass. See father class for parameters """
        out_dict = super().forward(x, *args, **kwargs)
        return out_dict

    def _get_predictor(self):
        """ Instanciating the prediction model """
        pred_model = models.LSTM if self.linear else models.ConvLSTM
        in_size = out_size = self.model_params["enc_dec"]["dim"]
        predictor = pred_model(
                input_size=in_size,
                output_size=out_size,
                hidden_size=self.model_params["LSTM"]["hidden_dim"],
                num_layers=self.model_params["LSTM"]["num_layers"],
            )
        return predictor


class SVG_LP(models.VideoPredModel):
    """ SVG-LP model intializer """

    def __init__(self, model_params, linear=True, **kwargs):
        """ Model intializer """
        super().__init__(model_params=model_params, linear=linear, **kwargs)

        self.linear = linear
        self.predictor = self._get_predictor()
        self.prior = self._get_prior_post(model_key="LSTM_Prior")
        self.posterior = self._get_prior_post(model_key="LSTM_Posterior")
        return

    def forward(self, x, *args, **kwargs):
        """ Forward pass. See father class for parameters """
        out_dict = super().forward(x, *args, **kwargs)
        return out_dict

    def _get_predictor(self):
        """ Instanciating the prediction model """
        out_size = self.model_params["enc_dec"]["dim"]
        in_size = out_size + self.model_params["LSTM_Prior"]["latent_dim"]
        pred_model = models.LSTM if self.linear else models.ConvLSTM
        predictor = pred_model(
                input_size=in_size,
                output_size=out_size,
                hidden_size=self.model_params["LSTM"]["hidden_dim"],
                num_layers=self.model_params["LSTM"]["num_layers"],
            )
        return predictor

    def _get_prior_post(self, model_key="LSTM_Prior"):
        """
        Instanciating the prior or posterior model

        Args:
        -----
        model_key: string
            key of the model parameters for prior or posterior
        """
        assert model_key in ["LSTM_Prior", "LSTM_Posterior"]
        prior_model = models.Gaussian_LSTM if self.linear else models.GaussianConvLSTM
        model = prior_model(
                input_size=self.model_params["enc_dec"]["dim"],
                output_size=self.model_params["LSTM_Prior"]["latent_dim"],
                hidden_size=self.model_params[model_key]["hidden_dim"],
                num_layers=self.model_params[model_key]["num_layers"]
            )
        return model


#
