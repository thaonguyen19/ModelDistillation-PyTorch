import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['ResNet', 'resnet18', 'rn_builder']


class PassZeros(nn.Module):
    def __init__(self, out_size, conv):
        super(PassZeros, self).__init__()
        def set_and_register(k, v):
            v = torch.from_numpy(np.array(v, dtype=int))
            setattr(self, k, v)
            self.register_buffer(k, v)
        set_and_register('out_size', [out_size])
        set_and_register('padding', conv.padding)
        set_and_register('dilation', conv.dilation)
        set_and_register('kernel', conv.kernel_size)
        set_and_register('stride', conv.stride)
    def _get_size(self, H, P, K, S, D=1):
        return int((H + 2 * P - D * (K - 1) - 1) / float(S) + 1)
    def get_size(self, H, ind=0):
        P = self._buffers['padding'][ind]
        D = self._buffers['dilation'][ind]
        K = self._buffers['kernel'][ind]
        S = self._buffers['stride'][ind]
        return self._get_size(H, P, K, S, D)
    def forward(self, x):
        out_size = self._buffers['out_size'][0]
        s1 = self.get_size(x.size(2), ind=0)
        s2 = self.get_size(x.size(3), ind=1)
        ret = torch.cuda.FloatTensor(x.size(0), out_size, s1, s2)
        ret.zero_()
        return Variable(ret)

class ZeroPadBN(nn.Module):
    def __init__(self, indexes, bn):
        super(ZeroPadBN, self).__init__()
        if len(indexes) != 0:
            self.indexes = torch.from_numpy(np.array(indexes, dtype=int))
        else:
            self.indexes = None
        self.register_buffer('indexes', self.indexes)
        self.bn = bn
        self.init()

    def init(self):
        if 'indexes' in self._buffers and self._buffers['indexes'] is not None:
            self.set_ind = set(list(self._buffers['indexes'].numpy()))
        else:
            self.set_ind = set()

    def update_inds(self, new_indexes):
        assert len(new_indexes) != 0
        new_inds = torch.from_numpy(np.array(new_indexes, dtype=int))
        if self._buffers['indexes'] is None:
            self._buffers['indexes'] = new_inds
        else:
            cur_indexes = self._buffers['indexes']
            max_ind = max(max(new_inds), max(cur_indexes)) + \
                len(new_inds) + len(cur_indexes)
            zbn1 = ZeroPadBN(new_indexes, nn.Sequential())
            zbn2 = ZeroPadBN(cur_indexes.cpu().numpy(), nn.Sequential())
            inp = torch.ones(1, max_ind, 1, 1).cuda(async=True)
            tmp = zbn2.forward(zbn1.forward(inp)).squeeze().data.cpu().numpy()
            tmp = np.where(tmp == 0)[0]
            tmp = torch.from_numpy(np.array(tmp, dtype=int))
            self._buffers['indexes'] = tmp
        self.init()

    def forward(self, x):
        x = self.bn(x)
        if len(self.set_ind) == 0:
            return x

        num_filters_alive = x.size(1)
        num_zero_filters = len(self.set_ind)
        total_num_filters = num_filters_alive + num_zero_filters

        with_zeros = Variable(torch.cuda.FloatTensor(x.size(0), total_num_filters, x.size(2), x.size(3)).zero_())
        alive_filters_indices = [i for i in range(total_num_filters) if i not in self.set_ind]
        with_zeros.index_copy_(1, Variable(torch.LongTensor(alive_filters_indices)).cuda(async=True), x)

        return with_zeros


class MyModuleList(nn.ModuleList):
    def __add__(self, x):
        tmp = [m for m in self.modules()] + [m for m in x.modules()]
        return MyModuleList(tmp)
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

def make_basic_block(inplanes, planes, stride=1, downsample=None):
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    block_list = MyModuleList([
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
    ])
    if downsample == None:
        residual = MyModuleList([])
    else:
        residual = downsample
    return (block_list, residual)

def make_bottleneck_block(inplanes, planes, stride=1, downsample=None):
    block_list = MyModuleList([
            # conv bn relu
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            # conv bn relu
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            # conv bn
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
    ])
    if downsample == None:
        residual = MyModuleList([])
    else:
        residual = downsample
    return (block_list, residual)

class ResNet(nn.Module):
    def __init__(self, section_reps,
                 num_classes=1000, nbf=64,
                 conv1_size=7, conv1_pad=3,
                 downsample_start=True,
                 use_basic_block=True,
                 train_death_rate=None,
                 test_death_rate=None):
        super(ResNet, self).__init__()

        if train_death_rate == None:
            self.train_death_rate = [[0.0] * x for x in section_reps]
        else:
            self.train_death_rate = train_death_rate
        if test_death_rate == None:
            self.test_death_rate = [[0.0] * x for x in section_reps]
        else:
            self.test_death_rate = test_death_rate
        if not all(map(lambda i: len(self.train_death_rate[i]) == section_reps[i],
                       range(len(section_reps)))):
            raise Exception('Train death rates do not match size')
        if not all(map(lambda i: len(self.test_death_rate[i]) == section_reps[i],
                       range(len(section_reps)))):
            raise Exception('Test death rates do not match size')

        train_total_dr = sum(map(sum, self.train_death_rate))
        test_total_dr = sum(map(sum, self.test_death_rate))
        self.pad_shortcut = (train_total_dr + test_total_dr) != 0 # FIXME
        if use_basic_block:
            self.expansion = 1
            self.block_fn = make_basic_block
        else:
            self.expansion = 4
            self.block_fn = make_bottleneck_block
        self.downsample_start = downsample_start
        self.inplanes = nbf

        self.conv1 = nn.Conv2d(3, nbf, kernel_size=conv1_size,
                               stride=downsample_start + 1, padding=conv1_pad, bias=False)
        self.bn1 = nn.BatchNorm2d(nbf)
        self.sections = []
        for i, section_rep in enumerate(section_reps):
            self.sections.append(self._make_section(nbf * (2 ** i), section_rep, stride=(i != 0) + 1))
        lin_inp = nbf * int(2 ** (len(section_reps) - 1)) * self.expansion \
            if len(self.sections) != 0 else nbf
        self.fc = nn.Linear(lin_inp, num_classes)

        self.update_modules()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def update_modules(self):
        # PyTorch requires the layers to be registered for propogation purposes.
        # If we ever change the layers, everything goes to shit. So update it
        self.registered = MyModuleList([])
        for section in self.sections:
            for block, shortcut in section:
                self.registered.append(block)
                self.registered.append(shortcut)

    def _make_section(self, planes, num_blocks, stride=1):
        if stride != 1 or self.inplanes != planes * self.expansion:
            # if False and self.pad_shortcut:
            #     downsample = MyModuleList([nn.AvgPool2d(stride), ZeroPad(stride)])
            # else:
            downsample = MyModuleList([
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * self.expansion),
            ])
        else:
            downsample = None

        blocks = []
        blocks.append(self.block_fn(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.expansion
        for i in range(1, num_blocks):
            blocks.append(self.block_fn(self.inplanes, planes))

        return blocks

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample_start:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        death_rates = self.train_death_rate if self.training else self.test_death_rate
        for sec_ind, section in enumerate(self.sections):
            for block_ind, (block, shortcut) in enumerate(section):
                dr = death_rates[sec_ind][block_ind]
                x_input = x
                if len(shortcut) != 0:
                    x = shortcut(x)
                if dr == 0 or torch.rand(1)[0] >= dr:
                    x_conv = block(x_input)
                    if self.training:
                        x_conv /= (1. - dr)
                    x = x + x_conv
                    x = F.relu(x)

        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Only basic block for now
def rn_builder(section_reps, **kwargs):
    return ResNet(section_reps, **kwargs)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([2, 2, 2, 2], **kwargs)
    return model
