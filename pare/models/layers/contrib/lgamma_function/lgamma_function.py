from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from scipy.special import digamma, gammaln
from torch.autograd import Function


class LogGammaFunction(Function):
    def forward(self, inputs):
        self.save_for_backward(inputs)
        outputs = inputs.clone()
        if inputs.is_cuda:
            outputs[:] = torch.from_numpy(gammaln(inputs.cpu().numpy().astype(np.float64)).astype(np.float32)).cuda()
        else:
            outputs[:] = torch.from_numpy(gammaln(inputs.numpy().astype(np.float64)).astype(np.float32))
        return outputs

    def backward(self, grad_output):
        inputs = self.saved_tensors[0]
        grad = grad_output.clone()
        if grad_output.is_cuda:
            grad[:] = torch.from_numpy(digamma(inputs.cpu().numpy().astype(np.float64)).astype(np.float32)).cuda()
        else:
            grad[:] = torch.from_numpy(digamma(inputs.numpy().astype(np.float64)).astype(np.float32))
        grad_input = grad_output * grad
        return grad_input


def lgamma(inputs):
    return LogGammaFunction()(inputs)


def lbeta(inputs):
    log_prod_gamma_x = torch.sum(lgamma(inputs), dim=-1, keepdim=True)
    sum_x = torch.sum(inputs, dim=-1, keepdim=True)
    log_gamma_sum_x = lgamma(sum_x)
    result = log_prod_gamma_x - log_gamma_sum_x
    return result
