# Lipschitz Multiscale Deep Equilibrium Models: A Theoretically Guaranteed and Accelerated Approach @AISTATS2026
Code for reproducing experiments in our paper.  
Our experiments were based on the implementation of [MDEQ](https://github.com/locuslab/deq/tree/master/MDEQ-Vision).

# Abstract
Deep equilibrium models (DEQs) achieve infinitely deep network representations without stacking layers by exploring fixed points of layer transformations in neural networks.
Such models constitute an innovative approach that achieves performance comparable to state-of-the-art methods in many large-scale numerical experiments, despite requiring significantly less memory.
However, DEQs face the challenge of requiring vastly more computational time for training and inference than conventional methods, as they repeatedly perform fixed-point iterations with no convergence guarantee upon each input. Therefore, this study explored an approach to improve fixed-point convergence and consequently reduce computational time by restructuring the model architecture to guarantee fixed-point convergence.
Our proposed approach for image classification, Lipschitz multiscale DEQ, has theoretically guaranteed fixed-point convergence for both forward and backward passes by hyperparameter adjustment, achieving up to a 4.75$\times$ speedup in numerical experiments on CIFAR-10 at the cost of a minor drop in accuracy.

# Wandb Setup
Please change entity name `XXXXXX` to your wandb entitiy in ./MDEQ-Vision/tools/cls_train.py.
```
parser.add_argument("--wandb_entity", type=str, default='XXXXXX', help='entity of wandb team')
```

# Usage
Please edit config file, i.e., ./MDEQ-Vision/experiments/cifar/cls_mdeq_LARGE.yaml.  
The following arguments were added by us.
```
CONTRACTIVE: 'l2'            #This is convolution constraint. The choices are 'none' and 'l2'.
TARGET_NORM: 2.0             #This is target norm for convolution operation, i.e., L_{Conv}^\star. Any number greater than zero can be specified.
ACTIVATION: 'ReLU'           #This is activation function. 
RELU_SLOPE: 0.4              #This is a slope of ReLU function. Any number greater than 0 and less than or equal to 1 can be specified. When set to 1, it matches ReLU.
RESIDUAL_ALPHA: 0.5          #This is convex combination parameter in residual block. Any number between 0 and 1 can be specified. When set to 0, residual connections are used.
FUSION_BETA: 0.3             #This is convex combination parameter in fusion layer. Any number between 0 and 1 can be specified. When set to 0, residual connections are used.
USE_STD_IN_GN: false         #In the case of true, group normalization is used; otherwise, mean-only group normalization is used.
GN_GAMMA_CLIP: 1.0           #This is the upper limit for the affine parameter \gamma in the normalization operation, i.e., L_{MGN}.
USE_SOFTMAX: true            #In the case of true, the weighted average is used in fusion layer; otherwise, a simple sum is used.
BACKWARD_METHOD: 'implicit'  #The choices are 'implicit', 'phantom_gradient', and 'jfb'.
```

Once you've finished editing the config file, one can run it with the following command.
```
python3 ./tools/cls_train.py --cfg ./experiments/cifar/cls_mdeq_LARGE.yaml
```
