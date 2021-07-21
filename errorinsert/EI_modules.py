# Copy from reliability

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np


__all__ = ['Conv2dEI', 'LinearEI']


def get_default_kwargs(kwargs):
    default = {
        'tmr': False,
        'ei_prob': 1e-3,
        'nbits': 8,

        'nw': -1,
        'ei_bits': False,

        'wg_flag': False,
        'bound1': 0,
        'bound2': 0,
        'wg_sort': False,

        'channel_flag': False,
        'tmr_array': 0
    }
    for k, v in default.items():
        if k not in kwargs:
            kwargs[k] = v
    return kwargs


def EI(x, ei_prob, inplace=False):
    if inplace:
        x_view = x.view(-1)
    else:
        x_view = x.clone().view(-1)
    # 1. generate random code
    total_bits = x.numel() * 8
    error_bits = int(total_bits * ei_prob)
    ei_position = random.sample(range(total_bits), error_bits)

    for ei_p in ei_position:
        idx_value = int(ei_p / 8)
        idx_bit = ei_p % 8
        x_view[idx_value] = reverse_bit(x_view[idx_value], idx_bit)
    return x_view.view(x.shape)

#当对channel-wise tmr配置时的注错方式，部分channel不注错就是保护
def EI_channel(x, ei_prob, tmr_array, channel_num, inplace=False):
    #print('tmr_array:{}'.format(tmr_array))
    if inplace:
        x_view = x.view(-1)
    else:
        x_view = x.clone().view(-1)
    total_bits = x.numel() * 8
    part = int(total_bits / channel_num)
    ei_bits = []
    #拼接要注错的channel的索引
    for i in range(channel_num):
        if tmr_array[i] == 1:
            continue
        for b in range(i*part, (i+1)*part):
            ei_bits.append(b)
    error_bits = int(len(ei_bits) * ei_prob)
    ei_position = random.sample(ei_bits, error_bits)

    for ei_p in ei_position:
        idx_value = int(ei_p / 8)
        idx_bit = ei_p % 8
        x_view[idx_value] = reverse_bit(x_view[idx_value], idx_bit)
    return x_view.view(x.shape)


def EI_w(x, ei_prob, bound1, bound2, sort=False, inplace=False):
    if inplace:
        x_view = x.view(-1)
    else:
        x_view = x.clone().view(-1)
    _, indices = torch.sort(x_view)  # 升序排序
    total_bits = int(bound2 - bound1) * 8
    error_bits = int(total_bits * ei_prob)
    ei_position = random.sample(range(total_bits), error_bits)

    for ei_p in ei_position:
        idx_value = int(ei_p / 8) + bound1
        if sort:
            idx_value = indices[idx_value]
        else:
            idx_value = int(ei_p / 8) + bound1
        idx_bit = ei_p % 8
        x_view[idx_value] = reverse_bit(x_view[idx_value], idx_bit)
    return x_view.view(x.shape)

def EI_bit(x, num, inplace=False):
    if inplace:
        x_view = x.view(-1)
    else:
        x_view = x.clone().view(-1)
    idx_value = int(num/8)
    idx_bit = num % 8
    x_view[idx_value] = reverse_bit(x_view[idx_value], idx_bit)
    return x_view.view(x.shape)

def reverse_bit(value, bit_position):
    bitmask = 2 ** bit_position
    if bit_position == 7:
        bitmask = - 2 ** bit_position
    value = int(value.item()) ^ int(bitmask)
    return value

class Conv2dEI(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(Conv2dEI, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs = get_default_kwargs(kwargs)
        self.tmr = self.kwargs['tmr']
        self.ei_prob = self.kwargs['ei_prob']
        self.nbits = self.kwargs['nbits']
        self.nw = self.kwargs['nw']
        self.ei_bits = self.kwargs['ei_bits']

        if self.nbits <= 0:
            self.register_buffer('radix_position', None)
            self.register_buffer('init_state', None)
        else:
            # (input, weight) 两个 radix_position
            self.register_buffer('radix_position', torch.zeros(2))
            self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.nbits <= 0:
            return self._conv_forward(x, self.weight)

        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1

        if self.init_state == 0:
            il = torch.log2(self.weight.abs().max()) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position[1].data.fill_(self.nbits - il)
            print('Initialize radix position of {} weight with {}'.format(
                self._get_name(), int(self.radix_position[1].item())))

            il = torch.log2(x.abs().max()) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position[0].data.fill_(self.nbits - il)
            print('Initialize radix position of {} input with {}'.format(
                self._get_name(), int(self.radix_position[0].item())))
            self.init_state.fill_(1)

        alpha_i = 2 ** self.radix_position[0]
        alpha_w = 2 ** self.radix_position[1]
        x_int = round_pass((x * alpha_i).clamp(Qn, Qp))
        w_int = round_pass((self.weight * alpha_w).clamp(Qn, Qp))

        x_int_ei = EI(x_int, self.ei_prob * 2)
        w_int_ei = EI(w_int, self.ei_prob)

        # tmr here

        x_ei = x_int_ei / alpha_i
        w_ei = w_int_ei / alpha_w
        return self._conv_forward(x_ei, w_ei)

    def extra_repr(self):
        s_prefix = super(Conv2dEI, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs)


class LinearEI(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(LinearEI, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs = get_default_kwargs(kwargs)
        self.tmr = self.kwargs['tmr']
        self.ei_prob = self.kwargs['ei_prob']
        self.nbits = self.kwargs['nbits']
        
        self.nw = self.kwargs['nw']
        self.ei_bits = self.kwargs['ei_bits']

        self.bound1 = self.kwargs['bound1']
        self.bound2 = self.kwargs['bound2']
        self.wg_flag = self.kwargs['wg_flag']
        self.wg_sort = self.kwargs['wg_sort']  # 是否排序后进行划分group
        self.channel_flag = self.kwargs['channel_flag']  #是否使用EI_channel
        self.tmr_array = self.kwargs['tmr_array']  #当使用EI_channel要传入的参数
        
        if self.nbits <= 0:
            self.register_buffer('radix_position', None)
            self.register_buffer('init_state', None)
        else:
            # (input, weight) 两个 radix_position
            self.register_buffer('radix_position', torch.zeros(2))
            self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.nbits <= 0:
            return F.linear(x, self.weight, self.bias)

        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0 and x.shape[0] >= 32:  # batch_size >= 32
            il = torch.log2(self.weight.abs().max()) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position[1].data.fill_(self.nbits - il)
            print('Initialize radix position of {} weight with {}'.format(
                self._get_name(), int(self.radix_position[1].item())))

            # batch size; remove sqrt[bs] outliers. topk
            batch_size = x.shape[0]
            il = torch.log2(x.abs().view(-1).topk(int(math.sqrt(batch_size)))[0][-1]) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position[0].data.fill_(self.nbits - il)
            print('Initialize radix position of {} input with {}'.format(
                self._get_name(), int(self.radix_position[0].item())))
            self.init_state.fill_(1)

        alpha_i = 2 ** self.radix_position[0]
        alpha_w = 2 ** self.radix_position[1]
        x_int = round_pass((x * alpha_i).clamp(Qn, Qp))
        w_int = round_pass((self.weight * alpha_w).clamp(Qn, Qp))

        if self.ei_bits and self.nw >= 0:
            w_int_ei = EI_bit(w_int, self.nw, self.ei_prob)
            w_ei = w_int_ei / alpha_w
            return F.linear(x_int / alpha_i, w_ei, self.bias)

        if self.wg_flag:
            w_int_ei = EI_w(w_int, self.ei_prob, self.bound1, self.bound2, self.wg_sort)
            w_ei = w_int_ei / alpha_w
            return F.linear(x_int / alpha_i, w_ei, self.bias)

        if self.channel_flag:
            w_int_ei = EI_channel(w_int, self.ei_prob, self.tmr_array, self.in_features)
            w_ei = w_int_ei / alpha_w
            return F.linear(x_int / alpha_i, w_ei, self.bias)

        x_int_ei = EI(x_int, self.ei_prob * 2)
        w_int_ei = EI(w_int, self.ei_prob)
        if self.tmr:
            x2 = EI(x_int, self.ei_prob * 2)
            x3 = EI(x_int, self.ei_prob * 2)
            x_int_ei = TMR(x_int_ei, x2, x3)
            w2 = EI(w_int, self.ei_prob)
            w3 = EI(w_int, self.ei_prob)
            w_int_ei = TMR(w_int_ei, w2, w3)
        x_ei = x_int_ei / alpha_i
        w_ei = w_int_ei / alpha_w
        return F.linear(x_ei, w_ei, self.bias)

    def set_ei_prob(self, value):
        self.ei_prob = value
        self.kwargs['ei_prob'] = value

    def set_tmr(self, value):
        self.tmr = value
        self.kwargs['tmr'] = value

    def set_ei_bits(self, value):
        self.ei_bits = value
        self.kwargs['ei_bits'] = value

    def set_wg_flag(self, value):
        self.wg_flag = value
        self.kwargs['wg_flag'] = value

    def set_wg_sort(self, value):
        self.wg_sort = value
        self.kwargs['wg_sort'] = value

    def set_nw(self, value):
        self.nw = value
        self.kwargs['nw'] = value

    def set_bound(self, value1, value2):
        self.bound1 = value1
        self.kwargs['bound1'] = value1
        self.bound2 = value2
        self.kwargs['bound2'] = value2

    def set_channel_flag(self, value):
        self.channel_flag = value
        self.kwargs['channel_flag'] = value

    def set_tmr_array(self, value):
        self.tmr_array = value
        self.kwargs['tmr_array'] = value

    def extra_repr(self):
        s_prefix = super(LinearEI, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs)


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad
