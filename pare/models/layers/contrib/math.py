from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from numbers import Number
import numpy as np


# Tested against Matlab: Works correctly!
def normcdf(value, mu=0.0, stddev=1.0):
    sinv = (1.0 / stddev) if isinstance(stddev, Number) else stddev.reciprocal()
    return 0.5 * (1.0 + torch.erf((value - mu) * sinv / np.sqrt(2.0)))


def _normal_log_pdf(value, mu, stddev):
    var = (stddev ** 2)
    log_scale = np.log(stddev) if isinstance(stddev, Number) else torch.log(stddev)
    return -((value - mu) ** 2) / (2.0*var) - log_scale - np.log(np.sqrt(2.0*np.pi))


# Tested against Matlab: Works correctly!
def normpdf(value, mu=0.0, stddev=1.0):
    return torch.exp(_normal_log_pdf(value, mu, stddev))
