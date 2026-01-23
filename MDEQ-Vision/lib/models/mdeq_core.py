from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools
from termcolor import colored

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._utils
import torch.nn.functional as F
import torch.autograd as autograd

sys.path.append("lib/")
from utils.utils import get_world_size, get_rank

sys.path.append("../")
from lib.optimizations import VariationalHidDropout2d, weight_norm
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate, power_method
from lib.layer_utils import list2vec, vec2list, norm_diff, conv3x3, conv5x5

from modules.deq2d import *
from modules.jfb import JFB_Function
from mdeq_forward_backward import MDEQWrapper

BN_MOMENTUM = 0.1
BLOCK_GN_AFFINE = True    # Don't change the value here. The value is controlled by the yaml files.
FUSE_GN_AFFINE = True     # Don't change the value here. The value is controlled by the yaml files.
POST_GN_AFFINE = True     # Don't change the value here. The value is controlled by the yaml files.
DEQ_EXPAND = 5        # Don't change the value here. The value is controlled by the yaml files.
NUM_GROUPS = 4        # Don't change the value here. The value is controlled by the yaml files.
logger = logging.getLogger(__name__)

class ScaledReLU(nn.Module):
    def __init__(self, slope=1.0, inplace=False):
        """A custom ReLU that scales the positive part by a slope."""
        super().__init__()
        self.slope = slope
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=self.inplace)

    def forward(self, x):
        return self.relu(x) * self.slope

    def extra_repr(self):
        return f'slope={self.slope}'

class CustomGroupNorm(nn.Module):
    """
    A customizable Group Normalization layer.
    Can be configured to perform standard GroupNorm or Mean-Only GroupNorm.
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, use_std=True):
        """
        Args:
            num_groups (int): Number of groups to separate the channels into.
            num_channels (int): Number of channels of the input tensor.
            eps (float): A value added to the denominator for numerical stability.
            affine (bool): If True, this module has learnable affine parameters.
            use_std (bool): If True, divide by the standard deviation (standard GroupNorm).
                            If False, only subtract the mean (Mean-Only GroupNorm).
        """
        super(CustomGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.use_std = use_std

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        N, C, H, W = x.shape
        
        # 1. Reshape for grouping
        x = x.view(N, self.num_groups, -1)
        
        # 2. Calculate mean
        mean = x.mean(dim=-1, keepdim=True)
        
        # 3. Center the data
        x_centered = x - mean
        
        # 4. (Optional) Calculate variance and normalize
        if self.use_std:
            # Standard GroupNorm path
            var = x_centered.var(dim=-1, keepdim=True)
            x_normalized = x_centered / torch.sqrt(var + self.eps)
        else:
            # Mean-Only GroupNorm path
            x_normalized = x_centered
            
        # 5. Reshape back to original shape
        x_out = x_normalized.view(N, C, H, W)
        
        # 6. Apply affine transformation
        if self.affine:
            x_out = x_out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            
        return x_out

class GroupSort(nn.Module):
    """
    Applies sorting independently to groups of channels.
    This is a 1-Lipschitz activation function w.r.t. the L2 norm.
    """
    def __init__(self, num_groups=1):
        super().__init__()
        if not isinstance(num_groups, int) or num_groups <= 0:
            raise ValueError("num_groups must be a positive integer.")
        self.num_groups = num_groups

    def forward(self, x):
        shape = x.shape
        if x.dim() <= 1:
            return x
        
        num_channels = shape[1]
        if num_channels % self.num_groups != 0:
            raise ValueError(f"Number of channels ({num_channels}) must be divisible by num_groups ({self.num_groups}).")
        
        group_size = num_channels // self.num_groups
        
        if group_size == 1:
            return x
            
        x_grouped = x.view(shape[0], self.num_groups, group_size, *shape[2:])
        x_sorted, _ = torch.sort(x_grouped, dim=2, descending=True)
        return x_sorted.view(shape)

class ScaledGroupSort(nn.Module):
    """
    Applies GroupSort and then scales the output by a learnable or fixed factor.
    The Lipschitz constant of this layer is exactly `abs(scale_factor)`.
    """
    def __init__(self, num_groups=1, scale_factor=1.0):
        """
        Args:
            num_channels (int): The number of channels of the input tensor.
            num_groups (int): The number of groups to sort within.
            scale_factor (float): The factor to scale the output. Must be non-negative.
        """
        super().__init__()
        if not isinstance(scale_factor, float) or not (0.0 <= scale_factor):
             raise ValueError("scale_factor must be a non-negative float.")
        
        self.group_sort = GroupSort(num_groups=num_groups)
        self.scale_factor = scale_factor

    def forward(self, x):
        sorted_x = self.group_sort(x)
            
        return sorted_x * self.scale_factor
            
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_big_kernels=0, dropout=0.0, wnorm=False, activation='ReLU', relu_slope=1.0, alpha=0.0, use_std=True):
        """
        A canonical residual block with two 3x3 convolutions and an intermediate ReLU. Corresponds to Figure 2
        in the paper.
        """
        super(BasicBlock, self).__init__()
        conv1 = conv5x5 if n_big_kernels >= 1 else conv3x3
        conv2 = conv5x5 if n_big_kernels >= 2 else conv3x3
        inner_planes = int(DEQ_EXPAND*planes)
        num_GS = 1

        self.conv1 = conv1(inplanes, inner_planes)
        self.gn1 = CustomGroupNorm(NUM_GROUPS, inner_planes, affine=BLOCK_GN_AFFINE, use_std=use_std)
        if activation == 'ReLU':
            self.relu = ScaledReLU(slope=relu_slope, inplace=True)
        else:
            self.relu = ScaledGroupSort(num_groups=num_GS, scale_factor=relu_slope)
        
        self.conv2 = conv2(inner_planes, planes)
        self.gn2 = CustomGroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE, use_std=use_std)

        self.gn3 = CustomGroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE, use_std=use_std)

        if activation == 'ReLU':
            self.relu3 = ScaledReLU(slope=relu_slope, inplace=True)
        else:
            self.relu3 = ScaledGroupSort(num_groups=num_GS, scale_factor=relu_slope)
            
        self.downsample = downsample
        self.drop = VariationalHidDropout2d(dropout)
        if wnorm: self._wnorm()
        self.alpha = alpha
    
    def _wnorm(self):
        """
        Register weight normalization
        """
        self.conv1, self.conv1_fn = weight_norm(self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(self.conv2, names=['weight'], dim=0)
    
    def _reset(self, bsz, d, H, W):
        """
        Reset dropout mask and recompute weight via weight normalization
        """
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)
        self.drop.reset_mask(bsz, d, H, W)
            
    def forward(self, x, injection=None):
        if injection is None: injection = 0
        residual = x

        out = self.relu(self.gn1(self.conv1(x)))
        out = self.drop(self.conv2(out)) + injection
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.alpha > 0:
            out = (1.0 - self.alpha) * residual + self.alpha * out
        else:
            out += residual
        out = self.gn3(self.relu3(out))
        return out
    
       
blocks_dict = { 'BASIC': BasicBlock }


class BranchNet(nn.Module):
    def __init__(self, blocks):
        """
        The residual block part of each resolution stream
        """
        super().__init__()
        self.blocks = blocks
    
    def forward(self, x, injection=None):
        blocks = self.blocks
        y = blocks[0](x, injection)
        for i in range(1, len(blocks)):
            y = blocks[i](y)
        return y

class DownsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res, activation='ReLU', relu_slope=1.0, use_std=True):
        """
        A downsample step from resolution j (with in_res) to resolution i (with out_res). A series of 2-strided convolutions.
        """
        super(DownsampleModule, self).__init__()
        # downsample (in_res=j, out_res=i)
        convs = []
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = out_res - in_res
        
        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False}
        for k in range(level_diff):
            intermediate_out = out_chan if k == (level_diff-1) else inp_chan
            components = [('conv', nn.Conv2d(inp_chan, intermediate_out, **kwargs)), 
                          ('gnorm', CustomGroupNorm(NUM_GROUPS, intermediate_out, affine=FUSE_GN_AFFINE, use_std=use_std))]
            if k != (level_diff-1):
                if activation == 'ReLU':
                    components.append(('relu', ScaledReLU(slope=relu_slope, inplace=True)))
                else:
                    components.append(('groupsort', ScaledGroupSort(num_groups=1, scale_factor=relu_slope)))
            convs.append(nn.Sequential(OrderedDict(components)))
        self.net = nn.Sequential(*convs)  
            
    def forward(self, x):
        return self.net(x)


class UpsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res, activation=None, relu_slope=None, use_std=True):
        """
        An upsample step from resolution j (with in_res) to resolution i (with out_res). 
        Simply a 1x1 convolution followed by an interpolation.
        """
        super(UpsampleModule, self).__init__()
        # upsample (in_res=j, out_res=i)
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = in_res - out_res
        
        self.net = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(inp_chan, out_chan, kernel_size=1, bias=False)),
                        ('gnorm', CustomGroupNorm(NUM_GROUPS, out_chan, affine=FUSE_GN_AFFINE, use_std=use_std)),
                        ('upsample', nn.Upsample(scale_factor=2**level_diff, mode='nearest'))]))
        
    def forward(self, x):
        return self.net(x)

    
class MDEQModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels, big_kernels, dropout=0.0, activation='ReLU', relu_slope=1.0, alpha=0.0, beta=0.0, use_std=True, use_softmax=True):
        """
        An MDEQ layer (note that MDEQ only has one layer). 
        """
        super(MDEQModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_channels, big_kernels)

        self.num_branches = num_branches
        self.num_channels = num_channels
        self.big_kernels = big_kernels
        self.activation = activation
        self.relu_slope = relu_slope
        self.alpha = alpha
        self.beta = beta
        self.use_std = use_std
        self.use_softmax = use_softmax

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels, big_kernels, dropout=dropout, activation='ReLU', relu_slope=self.relu_slope, alpha=self.alpha, use_std=self.use_std)
        self.fuse_layers = self._make_fuse_layers()
        if self.activation == 'ReLU':
            self.post_fuse_layers = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    ('relu', ScaledReLU(slope=self.relu_slope, inplace=False)),
                    ('conv', nn.Conv2d(num_channels[i], num_channels[i], kernel_size=1, bias=False)),
                    ('gnorm', CustomGroupNorm(NUM_GROUPS // 2, num_channels[i], affine=POST_GN_AFFINE, use_std=self.use_std))
                ])) for i in range(num_branches)])
        else:
            self.post_fuse_layers = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    ('groupsort', ScaledGroupSort(num_groups=1, scale_factor=self.relu_slope)),
                    ('conv', nn.Conv2d(num_channels[i], num_channels[i], kernel_size=1, bias=False)),
                    ('gnorm', CustomGroupNorm(NUM_GROUPS // 2, num_channels[i], affine=POST_GN_AFFINE, use_std=self.use_std))
                ])) for i in range(num_branches)])

    def _check_branches(self, num_branches, blocks, num_blocks, num_channels, big_kernels):
        """
        To check if the config file is consistent
        """
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if num_branches != len(big_kernels):
            error_msg = 'NUM_BRANCHES({}) <> BIG_KERNELS({})'.format(
                num_branches, len(big_kernels))
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _wnorm(self):
        """
        Apply weight normalization to the learnable parameters of MDEQ
        """
        self.post_fuse_fns = []
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._wnorm()
            conv, fn = weight_norm(self.post_fuse_layers[i].conv, names=['weight'], dim=0)
            self.post_fuse_fns.append(fn)
            self.post_fuse_layers[i].conv = conv
        
        # Throw away garbage
        torch.cuda.empty_cache()
        
    def _reset(self, xs):
        """
        Reset the dropout mask and the learnable parameters (if weight normalization is applied)
        """
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._reset(*xs[i].shape)
            if 'post_fuse_fns' in self.__dict__:
                self.post_fuse_fns[i].reset(self.post_fuse_layers[i].conv)    # Re-compute (...).conv.weight using _g and _v

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, big_kernels, stride=1, dropout=0.0, activation='ReLU', relu_slope=1.0, alpha=0.0, use_std=True):
        """
        Make a specific branch indexed by `branch_index`. This branch contains `num_blocks` residual blocks of type `block`.
        """
        layers = nn.ModuleList()
        n_channel = num_channels[branch_index]
        n_big_kernels = big_kernels[branch_index]
        for i in range(num_blocks[branch_index]):
            layers.append(block(n_channel, n_channel, n_big_kernels=n_big_kernels, dropout=dropout, activation=activation, relu_slope=relu_slope, alpha=alpha, use_std=use_std))
        return BranchNet(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, big_kernels, dropout=0.0, activation='ReLU', relu_slope=1.0, alpha=0.0, use_std=True):
        """
        Make the residual block (s; default=1 block) of MDEQ's f_\theta layer. Specifically,
        it returns `branch_layers[i]` gives the module that operates on input from resolution i.
        """
        branch_layers = [self._make_one_branch(i, block, num_blocks, num_channels, big_kernels, dropout=dropout, activation=activation, relu_slope=relu_slope, alpha=alpha, use_std=use_std) for i in range(num_branches)]
        return nn.ModuleList(branch_layers)

    def _make_fuse_layers(self):
        """
        Create the multiscale fusion layer (which does simultaneous up- and downsamplings).
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []                    # The fuse modules into branch #i
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)    # Identity if the same branch
                else:
                    module = UpsampleModule if j > i else DownsampleModule
                    fuse_layer.append(module(num_channels, in_res=j, out_res=i, activation=self.activation, relu_slope=self.relu_slope, use_std=self.use_std))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # fuse_layers[i][j] gives the (series of) conv3x3s that convert input from branch j to branch i
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_channels

    def forward(self, x, injection, *args):
        """
        The two steps of a multiscale DEQ module (see paper): a per-resolution residual block and 
        a parallel multiscale fusion step.
        """
        if injection is None:
            injection = [0] * len(x)
        if self.num_branches == 1:
            return [self.branches[0](x[0], injection[0])]

        # Step 1: Per-resolution residual block
        x_block = []
        for i in range(self.num_branches):
            x_block.append(self.branches[i](x[i], injection[i]))
        
        # Step 2: Multiscale fusion
        """
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            # Start fusing all #j -> #i up/down-samplings
            for j in range(self.num_branches):
                term = x_block[j] if i == j else self.fuse_layers[i][j](x_block[j])
                if self.beta > 0:
                    weight = (1.0 - self.beta) if i == j else self.beta / (self.num_branches-1)
                    y += weight * term
                else:
                    y += term
            x_fuse.append(self.post_fuse_layers[i](y))
        """
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            
            if self.beta > 0:
                if self.use_softmax:
                    y_self = (1.0 - self.beta) * x_block[i]

                    other_branch_terms = []
                    penalties = []
                    
                    for j in range(self.num_branches):
                        if i == j: continue 
                        
                        other_branch_terms.append(self.fuse_layers[i][j](x_block[j]))            
                        level_diff = self.fuse_layers[i][j].level_diff
                        penalty = float(level_diff) if j > i else 0.0
                        penalties.append(penalty)
                    
                    penalties_tensor = torch.tensor(penalties, device=x[0].device)
                    weights = F.softmax(-penalties_tensor, dim=0)
                    
                    y_others = 0
                    for k in range(len(other_branch_terms)):
                        y_others += weights[k] * other_branch_terms[k]
                    
                    y = y_self + self.beta * y_others

                else:
                    y = (1.0 - self.beta) * x_block[i]
                    other_branch_weight = self.beta / (self.num_branches - 1)
                    for j in range(self.num_branches):
                        if i != j:
                            y += other_branch_weight * self.fuse_layers[i][j](x_block[j])
            
            else:
                for j in range(self.num_branches):
                    y += x_block[j] if i == j else self.fuse_layers[i][j](x_block[j])
            
            x_fuse.append(self.post_fuse_layers[i](y))
            
        return x_fuse


class MDEQNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ model with the given hyperparameters

        Args:
            cfg ([config]): The configuration file (parsed from yaml) specifying the model settings
        """
        super(MDEQNet, self).__init__()
        global BN_MOMENTUM
        BN_MOMENTUM = kwargs.get('BN_MOMENTUM', 0.1)
        self.parse_cfg(cfg)
        init_chansize = self.init_chansize

        self.downsample = nn.Sequential(
            conv3x3(3, init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            ScaledReLU(slope=1.0, inplace=True),
            conv3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            ScaledReLU(slope=1.0, inplace=True))

        if self.downsample_times > 2:
            for i in range(3, self.downsample_times+1):
                self.downsample.add_module(f"DS{i}", conv3x3(init_chansize, init_chansize, stride=2))
                self.downsample.add_module(f"DS{i}-BN", nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True))
                self.downsample.add_module(f"DS{i}-RELU", ScaledReLU(slope=1.0, inplace=True))
                    
        
        # PART I: Input injection module
        if self.downsample_times == 0 and self.num_branches <= 2:
            # We use the downsample module above as the injection transformation
            self.stage0 = None
        else:
            self.stage0 = nn.Sequential(nn.Conv2d(self.init_chansize, self.init_chansize, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(self.init_chansize, momentum=BN_MOMENTUM, affine=True),
                                        ScaledReLU(slope=1.0, inplace=True))
        
        # PART II: MDEQ's f_\theta layer
        self.fullstage = self._make_stage(self.fullstage_cfg,
                                          self.num_channels,
                                          dropout=self.dropout,
                                          activation=self.activation,
                                          relu_slope=self.relu_slope,
                                          alpha=self.alpha,
                                          beta=self.beta,
                                          use_std=self.use_std,
                                          use_softmax=self.use_softmax)
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        if self.wnorm:
            self.fullstage._wnorm()

        if self.backward_method == 'phantom_gradient':
            print("INFO: Using Phantom Gradient for DEQ backward pass.")
            self.deq_solver_wrapper = MDEQWrapper(self.fullstage, 
                                                  tau=self.pg_tau, 
                                                  pg_steps=self.pg_steps)
        elif self.backward_method == 'jfb':
            print("INFO: Using Jacobian-Free Backpropagation.")
            self.deq_solver_wrapper = None
            
        else:
            print("INFO: Using Implicit Differentiation for DEQ backward pass.")
            self.deq_solver_wrapper = None 
        # ...

        self.iodrop = VariationalHidDropout2d(0.0)
        self.hook = None
        
    def parse_cfg(self, cfg):
        """
        Parse a configuration file
        """
        global DEQ_EXPAND, NUM_GROUPS, BLOCK_GN_AFFINE, FUSE_GN_AFFINE, POST_GN_AFFINE
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        self.init_chansize = self.num_channels[0]
        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.dropout = cfg['MODEL']['DROPOUT']
        self.wnorm = cfg['MODEL']['WNORM']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']
        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']   
        self.pretrain_steps = cfg['TRAIN']['PRETRAIN_STEPS']
        self.activation = cfg['MODEL']['ACTIVATION']
        self.relu_slope = cfg['MODEL']['RELU_SLOPE']
        self.alpha = cfg['MODEL']['RESIDUAL_ALPHA']
        self.beta = cfg['MODEL']['FUSION_BETA']
        self.use_std = cfg['MODEL']['USE_STD_IN_GN']
        self.use_softmax = cfg['MODEL']['USE_SOFTMAX']
        self.backward_method = cfg['MODEL']['BACKWARD_METHOD']
        self.pg_tau = cfg['MODEL']['PG_TAU']
        self.pg_steps = cfg['MODEL']['PG_STEPS']

        # DEQ related
        self.f_solver = eval(cfg['DEQ']['F_SOLVER'])
        self.b_solver = eval(cfg['DEQ']['B_SOLVER'])
        if self.b_solver is None:
            self.b_solver = self.f_solver
        self.f_thres = cfg['DEQ']['F_THRES']
        self.b_thres = cfg['DEQ']['B_THRES']
        self.stop_mode = cfg['DEQ']['STOP_MODE']
        
        # Update global variables
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
        BLOCK_GN_AFFINE = cfg['MODEL']['BLOCK_GN_AFFINE']
        FUSE_GN_AFFINE = cfg['MODEL']['FUSE_GN_AFFINE']
        POST_GN_AFFINE = cfg['MODEL']['POST_GN_AFFINE']
            
    def _make_stage(self, layer_config, num_channels, dropout=0.0, activation='ReLU', relu_slope=1.0, alpha=0.0, beta=0.0, use_std=True, use_softmax=True):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        block_type = blocks_dict[layer_config['BLOCK']]
        big_kernels = layer_config['BIG_KERNELS']
        return MDEQModule(num_branches, block_type, num_blocks, num_channels, big_kernels, dropout=dropout, activation=activation, relu_slope=relu_slope, alpha=alpha, beta=beta, use_std=use_std, use_softmax=use_softmax)

    def _forward(self, x, train_step=-1, compute_jac_loss=True, spectral_radius_mode=False, writer=None, **kwargs):
        """
        The core MDEQ module. In the starting phase, we can (optionally) enter a shallow stacked f_\theta training mode
        to warm up the weights (specified by the self.pretrain_steps; see below)
        """
        num_branches = self.num_branches
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        x = self.downsample(x)
        rank = get_rank()
        
        # Inject only to the highest resolution...
        x_list = [self.stage0(x) if self.stage0 else x]
        for i in range(1, num_branches):
            bsz, _, H, W = x_list[-1].shape
            x_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(x))   # ... and the rest are all zeros
            
        z_list = [torch.zeros_like(elem) for elem in x_list]

        z1 = list2vec(z_list)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]
        func = lambda z: list2vec(self.fullstage(vec2list(z, cutoffs), x_list))
        # For variational dropout mask resetting and weight normalization re-computations                                                                                 
        self.fullstage._reset(z_list)
        jac_loss = torch.tensor(0.0).to(x)
        sradius = torch.zeros(bsz, 1).to(x)
        deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)

        
        if self.training and self.backward_method == 'phantom_gradient':
            # ★★★ Phantom Gradient Path ★★★
            
            pg_steps = self.pretrain_pg_steps if train_step < self.pretrain_steps else self.pg_steps
            
            y_list = self.deq_solver_wrapper(z_list, x_list, 
                                             threshold=f_thres, 
                                             train_step=train_step,
                                             pg_steps=pg_steps,
                                             writer=writer)
            
            y_list = self.iodrop(y_list)
            return y_list, jac_loss.view(1,-1), sradius.view(1,-1)
        
        #z1 = list2vec(z_list)
        #cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]
        #func = lambda z: list2vec(self.fullstage(vec2list(z, cutoffs), x_list))
        
        # For variational dropout mask resetting and weight normalization re-computations
        #self.fullstage._reset(z_list)
        #jac_loss = torch.tensor(0.0).to(x)
        #sradius = torch.zeros(bsz, 1).to(x)
        deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)
        
        # Multiscale Deep Equilibrium!
        if not deq_mode:
            for layer_ind in range(self.num_layers): 
                z1 = func(z1)
            new_z1 = z1

            if self.training:
                if compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
                    jac_loss = jac_loss_estimate(new_z2, z2)

        else:
            with torch.no_grad():
                forward_phase = 'train_forward' if self.training else 'test_forward' #additional code
                result = self.f_solver(func, z1, threshold=f_thres, stop_mode=self.stop_mode, name="forward", phase=forward_phase) #additional
                z1 = result['result']
            new_z1 = z1

            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    new_z1 = func(z1.requires_grad_())
                _, sradius = power_method(new_z1, z1, n_iters=150)

            if self.training:
                if self.backward_method == 'jfb':
                    new_z1 = JFB_Function.apply(new_z1, x_list, self.fullstage, *self.fullstage.parameters())
                else:
                    new_z1 = func(z1.requires_grad_())
                    if compute_jac_loss:
                        jac_loss = jac_loss_estimate(new_z1, z1)
                    
                    def backward_hook(grad):
                        if self.hook is not None:
                            self.hook.remove()
                            torch.cuda.synchronize()
                        backward_phase = 'train_backward' #additional code  
                        result = self.b_solver(lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), 
                                               threshold=b_thres, stop_mode=self.stop_mode, phase=backward_phase)
                        return result['result']
                    self.hook = new_z1.register_hook(backward_hook)
                
        y_list = self.iodrop(vec2list(new_z1, cutoffs))
        return y_list, jac_loss.view(1,-1), sradius.view(-1,1)

    def forward(self, x, train_step=-1, **kwargs):
        raise NotImplemented    # To be inherited & implemented by MDEQClsNet and MDEQSegNet (see mdeq.py)
