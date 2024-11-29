from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from torch.autograd import Variable

class BayesLinear_Normalq(nn.Module):
    """Linear Layer where weights are sampled from a fully factorised Normal with learnable parameters. The likelihood
     of the weight samples under the prior and the approximate posterior are returned with each forward pass in order
     to estimate the KL term in the ELBO.
    """
    def __init__(self, n_in, n_out, prior_class):
        super(BayesLinear_Normalq, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior_class

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))

        self.lpw = 0
        self.lqw = 0

    def forward(self, X, sample=False):
        #         print(self.training)

        if not self.training and not sample:  # When training return MLE of w for quick validation
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0

        else:
            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch
            eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
            eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())

            # sample parameters
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

            W = self.W_mu + 1 * std_w * eps_W
            b = self.b_mu + 1 * std_b * eps_b

            output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)

            lqw = isotropic_gauss_loglike(W, self.W_mu, std_w) + isotropic_gauss_loglike(b, self.b_mu, std_b)
            lpw = self.prior.loglike(W) + self.prior.loglike(b)
            return output, lqw, lpw

class BayesConv_Normalq(nn.Module):
    """Convolutional Layer where weights are sampled from a fully factorised Normal with learnable parameters. The likelihood
        of the weight samples under the prior and the approximate posterior are returned with each forward pass in order
        to estimate the KL term in the ELBO.
        """
    def __init__(self, n_in_channels, n_out_channels, kernel_size, prior_class, stride=1,
                 padding=0, dilation=1):
        super(BayesConv_Normalq, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.prior = prior_class

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(torch.Tensor(n_out_channels, n_in_channels, *self.kernel_size).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(torch.Tensor(n_out_channels, n_in_channels, *self.kernel_size).uniform_(-3, -2))

        #self.b_mu = nn.Parameter(torch.Tensor(1, n_out_channels, 1, 1).uniform_(-0.1, 0.1))
        #self.b_p = nn.Parameter(torch.Tensor(1, n_out_channels, 1, 1).uniform_(-3, -2))
        self.b_mu = nn.Parameter(torch.Tensor(n_out_channels).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(n_out_channels).uniform_(-3, -2))

        self.out = lambda input, kernel, bias: F.conv2d(input, kernel, bias, self.stride, self.padding,
                                                       self.dilation, self.groups)

        self.lpw = 0
        self.lqw = 0

    def forward(self, X, sample=False):

        if not self.training and not sample:  # When training return MLE of w for quick validation
            output = self.out_bias(X, self.W_mu, self.b_mu)
            return output, 0, 0

        else:

            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch
            eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
            eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())

            # sample parameters
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

            W = self.W_mu + 1 * std_w * eps_W
            b = self.b_mu + 1 * std_b * eps_b

            output = self.out(X, W, b)

            lqw = isotropic_gauss_loglike(W, self.W_mu, std_w) + isotropic_gauss_loglike(b, self.b_mu, std_b)
            lpw = self.prior.loglike(W) + self.prior.loglike(b)
            return output, lqw, lpw

def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    cte_term = -(0.5) * np.log(2 * np.pi)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term)
    return out

class laplace_prior(object):
    def __init__(self, mu, b):
        self.mu = mu
        self.b = b

    def loglike(self, x, do_sum=True):
        if do_sum:
            return (-np.log(2 * self.b) - torch.abs(x - self.mu) / self.b).sum()
        else:
            return (-np.log(2 * self.b) - torch.abs(x - self.mu) / self.b)

class isotropic_gauss_prior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.cte_term = -(0.5) * np.log(2 * np.pi)
        self.det_sig_term = -np.log(self.sigma)

    def loglike(self, x, do_sum=True):

        dist_term = -(0.5) * ((x - self.mu) / self.sigma) ** 2
        if do_sum:
            return (self.cte_term + self.det_sig_term + dist_term).sum()
        else:
            return (self.cte_term + self.det_sig_term + dist_term)

class spike_slab_2GMM(object):
    def __init__(self, mu1, mu2, sigma1, sigma2, pi):
        self.N1 = isotropic_gauss_prior(mu1, sigma1)
        self.N2 = isotropic_gauss_prior(mu2, sigma2)

        self.pi1 = pi
        self.pi2 = (1 - pi)

    def loglike(self, x):
        N1_ll = self.N1.loglike(x)
        N2_ll = self.N2.loglike(x)

        # Numerical stability trick -> unnormalising logprobs will underflow otherwise
        max_loglike = torch.max(N1_ll, N2_ll)
        normalised_like = self.pi1 * torch.exp(N1_ll - max_loglike) + self.pi2 * torch.exp(N2_ll - max_loglike)
        loglike = torch.log(normalised_like) + max_loglike

        return loglike