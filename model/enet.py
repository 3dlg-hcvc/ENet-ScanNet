"""
Modified from https://github.com/daveredrum/D3Net/blob/main/model/enet.py
"""

import torch
import torch.nn as nn
from functools import reduce
from torch.autograd import Variable
import torch._utils

torch._utils._rebuild_tensor_v2


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, x):
        output = []
        for module in self._modules.values():
            output.append(module(x))
        return output if output else x


class Lambda(LambdaBase):
    def forward(self, x):
        return self.lambda_func(self.forward_prepare(x))


class LambdaMap(LambdaBase):
    def forward(self, x):
        # result is Variables list [Variable1, Variable2, ...]
        return list(map(self.lambda_func, self.forward_prepare(x)))


class LambdaReduce(LambdaBase):
    def forward(self, x):
        # result is a Variable
        return reduce(self.lambda_func, self.forward_prepare(x))


class Padding(nn.Module):
    # pad puts in [pad] amount of [value] over dimension [dim], starting at
    # index [index] in that dimension. If pad<0, index counts from the left.
    # If pad>0 index counts from the right.
    # When nInputDim is provided, inputs larger than that value will be considered batches
    # where the actual dim to be padded will be dimension dim + 1.
    def __init__(self, dim, pad, value, index, nInputDim):
        super(Padding, self).__init__()
        self.value = value
        # self.index = index
        self.dim = dim
        self.pad = pad
        self.nInputDim = nInputDim
        if index != 0:
            raise NotImplementedError("Padding: index != 0 not implemented")

    def forward(self, x):
        dim = self.dim
        if self.nInputDim != 0:
            dim += x.dim() - self.nInputDim
        pad_size = list(x.size())
        pad_size[dim] = self.pad
        padder = Variable(x.data.new(*pad_size).fill_(self.value))

        if self.pad < 0:
            padded = torch.cat((padder, x), dim)
        else:
            padded = torch.cat((x, padder), dim)
        return padded


# class Dropout(nn.Dropout):
#     """
#     Cancel out PyTorch rescaling by 1/(1-p)
#     """
#     def forward(self, input):
#         input = input * (1 - self.p)
#         return super(Dropout, self).forward(input)


class Dropout2d(nn.Dropout2d):
    """
    Cancel out PyTorch rescaling by 1/(1-p)
    """

    def forward(self, x):
        x = x * (1 - self.p)
        return super(Dropout2d, self).forward(x)


class StatefulMaxPool2d(nn.MaxPool2d):  # object keeps indices and input sizes
    def __init__(self, *args, **kwargs):
        super(StatefulMaxPool2d, self).__init__(*args, **kwargs)
        self.indices = None
        self.input_size = None

    def forward(self, x):
        return_indices, self.return_indices = self.return_indices, True
        output, indices = super(StatefulMaxPool2d, self).forward(x)
        self.return_indices = return_indices
        self.indices = indices
        self.input_size = x.size()
        if return_indices:
            return output, indices
        return output


def create_enet():
    enet = nn.Sequential(  # Sequential,
        LambdaMap(lambda x: x,  # ConcatTable,
                  nn.Conv2d(3, 13, (3, 3), (2, 2), (1, 1), (1, 1), 1),
                  StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False),
                  ),
        LambdaReduce(lambda x, y: torch.cat((x, y), 1)),
        nn.BatchNorm2d(16, 0.001, 0.1, True),
        nn.PReLU(16),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(16, 16, (2, 2), (2, 2), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                          StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False),
                          Padding(0, 48, 0, 0, 3),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 32, (2, 2), (2, 2), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                          StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False),
                          Padding(0, 64, 0, 0, 3),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (2, 2), (2, 2), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False),
                          nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (4, 4), (4, 4), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (8, 8), (8, 8), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False),
                          nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (16, 16), (16, 16), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (2, 2), (2, 2), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False),
                          nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (4, 4), (4, 4), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (8, 8), (8, 8), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False),
                          nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (16, 16), (16, 16), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        # # ENCODER END (add classifier)
        # nn.Sequential(
        #     nn.Conv2d(128, 41, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False)
        # )
    )
    return enet
