import numpy as np
import torch
from ctypes import *
import g_conf
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats
import numpy as np

# float / int quantum
model_type = int

lib=CDLL('errorinsert/err.so')
insert_float=lib.insert_float
insert_float.restype=c_float

def round_pass(x):
    return x.round()
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

def generateInsertList(x, rate):
    sz=x.view(-1).size()[0]
    n=sz*rate
    rn=scipy.stats.poisson.rvs(n)
    rn=min(rn, sz)
    print('sz:',sz,' rn:',rn)
    return np.random.randint(0, sz, rn)

def insertError(input):
    input_copy = input.clone()
    if model_type == int:
        max_value=max(input.abs().max(),1e-5)
        input_copy *= 128/max_value
        input_copy.round_()

    if g_conf.EI_CONV_OUT>0:
        rawErrorList = generateInsertList(input_copy, g_conf.EI_CONV_OUT)
        for j in rawErrorList:
            input_copy.view(-1)[j] = insert_fault(input_copy.view(-1)[j].item())

    if model_type == int:
        input_copy *= max_value/128

    return input_copy

def insertError_fc(input):
    input_copy = input.clone()

    if model_type == int:
        max_value=max(input.abs().max(),1e-5)
        input_copy *= 128/max_value
        input_copy.round_()

    if g_conf.EI_FC_OUT>0:
        rawErrorList = generateInsertList(input_copy, g_conf.EI_FC_OUT)
        for j in rawErrorList:
            input_copy.view(-1)[j] = insert_fault(input_copy.view(-1)[j].item())
    
    if model_type == int:
        input_copy *= max_value/128

    return input_copy

class Conv2dEI(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dEI, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        #x_ei=x
        x_ei = insertError(x)
        w_ei = self.weight
        #w_ei = insertError(self.weight)
        return F.conv2d(x_ei, w_ei, None, self.stride,
                        self.padding, self.dilation, self.groups)

class LinearEI(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(LinearEI, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
    def forward(self, x):
        #x_ei = x
        x_ei = insertError_fc(x)
        w_ei = self.weight
        #w_ei = insertError_fc(self.weight)
        return F.linear(x_ei, w_ei, self.bias)

def reverse_bit(value, bit_position):
    bitmask = 2 ** bit_position
    if bit_position == 7:
        bitmask = - 2 ** bit_position
    value = int(value) ^ int(bitmask)
    return value

def insert_fault(data):
    # TODO: Other data types
    # int8
    if model_type==int:
        assert -128<=int(data)<=128
        #errorbit=np.random.randint(0, 8)
        errorbit=7
        try:
            return reverse_bit(data, errorbit)
        except Exception:
            return data

    # float32
    errorbit=np.random.randint(0,32)
    value = float(insert_float(c_float(data), errorbit))
    return value

