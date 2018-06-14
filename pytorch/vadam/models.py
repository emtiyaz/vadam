import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


############################
## Multi-Layer Perceptron ##
############################

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_func="relu"):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if output_size is not None:
            self.output_size = output_size
            self.squeeze_output = False
        else :
            self.output_size = 1
            self.squeeze_output = True
             
        # Set activation function
        if act_func == "relu":
            self.act = F.relu
        elif act_func == "tanh":
            self.act = F.tanh
        elif act_func == "sigmoid":
            self.act = F.sigmoid
         
        # Define layers
        if len(hidden_sizes) == 0:
            # Logitstic regression
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            # Neural network
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)
 
    def forward(self, x):
        out = x
        for layer in self.hidden_layers:
            out = self.act(layer(out))
        z = self.output_layer(out)
        if self.squeeze_output:
            z = torch.squeeze(z)
        return z


#############################
## Bayesian Neural Network ##
#############################

class BNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_func, prior_prec=1.0, prec_init=1.0):
        super(type(self), self).__init__()
        self.input_size = input_size
        sigma_prior = 1.0/math.sqrt(prior_prec)
        sigma_init = 1.0/math.sqrt(prec_init)
        if output_size:
            self.output_size = output_size
            self.squeeze_output = False
        else :
            self.output_size = 1
            self.squeeze_output = True
        self.act = F.tanh if act_func == "tanh" else F.relu
        if len(hidden_sizes) == 0:
            self.hidden_layers = []
            self.output_layer = StochasticLinear(self.input_size, self.output_size, sigma_prior = sigma_prior, sigma_init = sigma_init)
        else:
            self.hidden_layers = nn.ModuleList([StochasticLinear(in_size, out_size, sigma_prior = sigma_prior, sigma_init = sigma_init) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            self.output_layer = StochasticLinear(hidden_sizes[-1], self.output_size, sigma_prior = sigma_prior, sigma_init = sigma_init)

    def forward(self, x):
        x = x.view(-1,self.input_size)
        out = x
        for layer in self.hidden_layers:
            out = self.act(layer(out))
        logits = self.output_layer(out)
        if self.squeeze_output:
            logits = torch.squeeze(logits)
        return logits

    def kl_divergence(self):
        kl = 0
        for layer in self.hidden_layers:
            kl += layer.kl_divergence()
        kl += self.output_layer.kl_divergence()
        return(kl)


###############################################
## Gaussian Mean-Field Linear Transformation ##
###############################################

class StochasticLinear(nn.Module):
    """Applies a stochastic linear transformation to the incoming data: :math:`y = Ax + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.
    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, sigma_prior=1.0, sigma_init=1.0, bias=True):
        super(type(self), self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_prior = sigma_prior
        self.sigma_init = sigma_init
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_spsigma = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = True
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_spsigma = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias_mu.data.uniform_(-stdv, stdv)
        self.weight_spsigma.data.fill_(math.log(math.exp(self.sigma_init)-1))
        if self.bias is not None:
            self.bias_spsigma.data.fill_(math.log(math.exp(self.sigma_init)-1))

    def forward(self, input):
        epsilon_W = torch.normal(mean=torch.zeros_like(self.weight_mu), std=1.0)
        weight = self.weight_mu + F.softplus(self.weight_spsigma) * epsilon_W
        if self.bias is not None:
            epsilon_b = torch.normal(mean=torch.zeros_like(self.bias_mu), std=1.0)
            bias = self.bias_mu + F.softplus(self.bias_spsigma) * epsilon_b
        return F.linear(input, weight, bias)

    def _kl_gaussian(self, p_mu, p_sigma, q_mu, q_sigma):
        var_ratio = (p_sigma / q_sigma).pow(2)
        t1 = ((p_mu - q_mu) / q_sigma).pow(2)
        return 0.5 * torch.sum((var_ratio + t1 - 1 - var_ratio.log()))

    def kl_divergence(self):
        mu = self.weight_mu
        sigma = F.softplus(self.weight_spsigma)
        mu0 = torch.zeros_like(mu)
        sigma0 = torch.ones_like(sigma) * self.sigma_prior
        kl = self._kl_gaussian(p_mu = mu, p_sigma = sigma, q_mu = mu0, q_sigma = sigma0)
        if self.bias is not None:
            mu = self.bias_mu
            sigma = F.softplus(self.bias_spsigma)
            mu0 = torch.zeros_like(mu)
            sigma0 = torch.ones_like(sigma) * self.sigma_prior
            kl += self._kl_gaussian(p_mu = mu, p_sigma = sigma, q_mu = mu0, q_sigma = sigma0)
        return kl

    def extra_repr(self):
        return 'in_features={}, out_features={}, sigma_prior={}, sigma_init={}, bias={}'.format(
            self.in_features, self.out_features, self.sigma_prior, self.sigma_init, self.bias is not None
        )

