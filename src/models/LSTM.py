"""
Recurrent models for frame prediction
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    """
    Implementation of an LSTM block: FC + LSTM + FC + Tan

    Args:
    -----
    input_size: integer
        dimensionality of the input vector. Number of input neurons in first FC
    outputs_size: integer
        dimensionality of the output vector. Number of output neurons in last FC
    hidden_size: integer
        dimensionality of the LSTM state and input
    num_layers: integer
        number of cascaded LSTM cells
    batch_size: integer/None
        number of elements in a mini-batch. Needed to initialize hidden state
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_size=None,
                 use_output=True, **kwargs):
        """ Initializer of the LSTM block """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # firing frequency parameters
        self.period = kwargs.get("period", 1)
        self.last_output = None
        self.counter = 0

        # modules
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.num_layers)])
        if use_output:
            self.output = nn.Sequential(
                    nn.Linear(hidden_size, output_size),
                    nn.Tanh())
        else:
            self.output = nn.Identity()

        self.hidden = self.init_hidden(batch_size) if batch_size is not None else None
        return

    def init_hidden(self, batch_size=1, device=None, **kwargs):
        """ Initializing hidden state vectors with zeros """
        hidden = []
        for i in range(self.num_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size)),
                           Variable(torch.zeros(batch_size, self.hidden_size))))
        if device is not None:
            hidden = [(h[0].to(device), h[1].to(device)) for h in hidden]
        self.hidden = hidden
        return hidden

    def forward(self, input, hidden_state=None):
        """ Forward pass through LSTM block """
        # Checking if it is time for this LSTM to fire
        should_fire = self.check_counters()
        if (not should_fire):
            return self.last_output

        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.num_layers):
            hidden_state = hidden_state if hidden_state is not None and i == 0 else self.hidden[i]
            self.hidden[i] = self.lstm[i](h_in, hidden_state)
            h_in = self.hidden[i][0]
        out = self.output(h_in)
        self.last_output = out
        return out

    def check_counters(self):
        """ Checking if it is firing time (c=0), updating/resetting counter """
        should_fire = (self.counter == 0)
        if(should_fire):
            self.reset_counter()
        else:
            self.counter = self.counter - 1
        return should_fire

    def reset_counter(self):
        """ Resetting period counter """
        self.counter = self.period - 1
        return

    def init_counter(self):
        """ Resetting period counter """
        self.counter = 0
        return


class Gaussian_LSTM(nn.Module):
    """
    LSTM, but instead of predicting the image features at the next frame, it models
    the underlying prior distribution of the latent variables in a sort of VAE fashion.

    See parameters from LSTM class
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_size=None, **kwargs):
        """ Initializer of the LSTM block """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # firing frequency parameters
        self.period = kwargs.get("period", 1)
        self.counter = 0
        self.last_output = None

        # modules
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.num_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden(batch_size) if batch_size is not None else None
        return

    def init_hidden(self, batch_size=1, device=None, **kwargs):
        """ Initializing hidden state vectors with zeros """
        hidden = []
        for i in range(self.num_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size)),
                           Variable(torch.zeros(batch_size, self.hidden_size))))
        if(device is not None):
            hidden = [(h[0].to(device), h[1].to(device)) for h in hidden]
        self.hidden = hidden
        return hidden

    def reparameterize(self, mu, logvar):
        """ Reparameterization trick """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, input):
        """ Forward pass through LSTM block """
        should_fire = self.check_counters()
        if (not should_fire):
            return self.last_output, should_fire

        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.num_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        self.last_output = (z, mu, logvar)
        return (z, mu, logvar), should_fire

    def check_counters(self):
        """ Checking if it is firing time (c=0), updating/resetting counter """
        should_fire = (self.counter == 0)
        if(should_fire):
            self.reset_counter()
        else:
            self.counter = self.counter - 1
        return should_fire

    def reset_counter(self):
        """ Resetting period counter """
        self.counter = self.period - 1
        return

    def init_counter(self):
        """ Resetting period counter """
        self.counter = 0
        return

#
