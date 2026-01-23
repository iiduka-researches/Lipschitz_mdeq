# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import sys
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mdeq_core import CustomGroupNorm

from core.cls_evaluate import accuracy
sys.path.append("../")
from utils.utils import save_checkpoint, AverageMeter
import random
from tqdm import tqdm


logger = logging.getLogger(__name__)

def log_conv_lipschitz(model):
    """
    Calculates and logs the L_inf Lipschitz constants of Conv2d layers in the model.
    """
    lipschitz_constants = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                W = module.weight.data
                abs_row_sum = torch.sum(torch.abs(W), dim=(1, 2, 3))
                lipschitz_constants.append(torch.max(abs_row_sum))
    
    if lipschitz_constants:
        lipschitz_tensor = torch.tensor(lipschitz_constants)
        
        wandb.log({
            'lipschitz/conv_inf_max': torch.max(lipschitz_tensor).item(),
            'lipschitz/conv_inf_mean': torch.mean(lipschitz_tensor).item(),
            'lipschitz/conv_inf_min': torch.min(lipschitz_tensor).item()
        })

def spectral_norm_conv_and_log(module, n_power_iterations=1, target_norm=1.0, device=None):
    """
    Applies spectral normalization to a conv layer's weight to cap its L2 norm
    at target_norm, and logs the original spectral norm to wandb.
    """
    W = module.weight.data
    device = W.device if device is None else device
    C_out, C_in, K, K = W.shape
    W_mat = W.view(C_out, -1)
    
    if not hasattr(module, 'u_for_spectral_norm'):
        u = torch.randn(C_out, device=W.device)
        u = F.normalize(u, dim=0)
        module.register_buffer('u_for_spectral_norm', u)
    else:
        u = module.u_for_spectral_norm

    with torch.no_grad():
        for _ in range(n_power_iterations):
            v = F.normalize(torch.mv(W_mat.t(), u), dim=0)
            u = F.normalize(torch.mv(W_mat, v), dim=0)

        sigma = torch.dot(u, torch.mv(W_mat, v))
        factor = torch.max(torch.tensor(1.0, device=device), sigma / float(target_norm))
        if factor > 1.0:
            module.weight.data /= factor
        module.u_for_spectral_norm.copy_(u)
    return sigma.item()

def log_gn_affine_params(model):
    """
    Calculates and logs the statistics of affine parameters (gamma/weight and beta/bias)
    in GroupNorm and CustomGroupNorm layers.
    """
    target_class_names = ["GroupNorm", "CustomGroupNorm"]
    
    gammas = []
    betas = [] 
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if module.__class__.__name__ in target_class_names:
                if module.weight is not None:
                    gammas.append(module.weight.data)
                if module.bias is not None:
                    betas.append(module.bias.data)
                    
    if gammas:
        all_gammas = torch.cat(gammas)
        wandb.log({
            'gamma/gn_max': torch.max(all_gammas).item(),
            'gamma/gn_mean': torch.mean(all_gammas).item(),
            'gamma/gn_min': torch.min(all_gammas).item(),
            'gamma/gn_mean_abs': torch.mean(torch.abs(all_gammas)).item()
        })
        
    if betas:
        all_betas = torch.cat(betas)
        wandb.log({
            'beta/gn_max': torch.max(all_betas).item(),
            'beta/gn_mean': torch.mean(all_betas).item(),
            'beta/gn_min': torch.min(all_betas).item(),
            'beta/gn_mean_abs': torch.mean(torch.abs(all_betas)).item()
        })

def train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict=None, topk=(1,5), contractive=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    jac_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    writer = writer_dict['writer'] if writer_dict else None
    global_steps = writer_dict['train_global_steps']
    update_freq = config.LOSS.JAC_INCREMENTAL

    forward_time = AverageMeter()
    backward_time = AverageMeter()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # switch to train mode
    model.train()

    epoch_start_time = time.time()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    for i, (input, target) in enumerate(train_loader):
        # train on partial training data
        if i >= effec_batch_num: break
            
        # measure data loading time
        data_time.update(time.time() - end)

        # compute jacobian loss weight (which is dynamically scheduled)
        deq_steps = global_steps - config.TRAIN.PRETRAIN_STEPS
        if deq_steps < 0:
            # We can also regularize output Jacobian when pretraining
            factor = config.LOSS.PRETRAIN_JAC_LOSS_WEIGHT
        elif epoch >= config.LOSS.JAC_STOP_EPOCH:
            # If are above certain epoch, we may want to stop jacobian regularization training
            # (e.g., when the original loss is 0.01 and jac loss is 0.05, the jacobian regularization
            # will be dominating and hurt performance!)
            factor = 0
        else:
            # Dynamically schedule the Jacobian reguarlization loss weight, if needed
            factor = config.LOSS.JAC_LOSS_WEIGHT + 0.1 * (deq_steps // update_freq)
        compute_jac_loss = (torch.rand([]).item() < config.LOSS.JAC_LOSS_FREQ) and (factor > 0)
        delta_f_thres = torch.randint(-config.DEQ.RAND_F_THRES_DELTA,2,[]).item() if (config.DEQ.RAND_F_THRES_DELTA > 0 and compute_jac_loss) else 0
        f_thres = config.DEQ.F_THRES + delta_f_thres
        b_thres = config.DEQ.B_THRES

        start_event.record() 
        
        output, jac_loss, _ = model(input, train_step=(lr_scheduler._step_count-1), 
                                    compute_jac_loss=compute_jac_loss,
                                    f_thres=f_thres, b_thres=b_thres, writer=writer)
        target = target.cuda(non_blocking=True)
        loss = criterion(output, target)
        jac_loss = jac_loss.mean()

        end_event.record()
        torch.cuda.synchronize()
        forward_time.update(start_event.elapsed_time(end_event))

        # compute gradient and do update step
        optimizer.zero_grad()

        start_event.record()
        
        if factor > 0:
            (loss + factor*jac_loss).backward()
        else:
            loss.backward()

        end_event.record() 
        torch.cuda.synchronize() 
        backward_time.update(start_event.elapsed_time(end_event))
        
        if config.TRAIN.CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP)
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()

        if config.MODEL.CONTRACTIVE == 'l2':
            l2_norms_before_norm = []
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, nn.Conv2d):
                        """
                        if config.MODEL.CONTRACTIVE == 'linf':
                            W = module.weight.data
                            abs_total_sum = torch.sum(torch.abs(W), dim=(1, 2, 3))
                            target_lipschitz = 1.0 
                            scale = torch.max(target_lipschitz * torch.ones_like(abs_total_sum), abs_total_sum)
                            module.weight.data /= scale.view(-1, 1, 1, 1)
                        """

                        #if config.MODEL.CONTRACTIVE == 'l2':
                        sigma = spectral_norm_conv_and_log(module, n_power_iterations=1, target_norm=config.MODEL.TARGET_NORM)
                        #l2_norms_before_norm.append(sigma)

        gamma_clip_val = config.MODEL.GN_GAMMA_CLIP
        if gamma_clip_val > 0:
            target_class_names = ["GroupNorm", "CustomGroupNorm"]
            with torch.no_grad():
                for name, module in model.named_modules():
                    if module.__class__.__name__ in target_class_names:
                        if module.weight is not None:
                            module.weight.data.clamp_(-gamma_clip_val, gamma_clip_val)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if compute_jac_loss:
            jac_losses.update(jac_loss.item(), input.size(0))

        prec1, prec5 = accuracy(output, target, topk=topk)
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}] ({3})\t' \
                  'Time {batch_time.avg:.3f}s\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.avg:.3f}s\t' \
                  'Loss {loss.avg:.5f}\t' \
                  'Jac (gamma) {jac_losses.avg:.4f} ({factor:.4f})\t' \
                  'Acc@1 {top1.avg:.3f}\t'.format(
                      epoch, i, effec_batch_num, global_steps, batch_time=batch_time,
                      speed=input.size(0)/batch_time.avg,
                      data_time=data_time, loss=losses, jac_losses=jac_losses, factor=factor, top1=top1)
            if 5 in topk:
                msg += 'Acc@5 {top5.avg:.3f}\t'.format(top5=top5)
            logger.info(msg)

            #log_conv_lipschitz(model)
            #log_gn_affine_params(model)

            #if config.MODEL.CONTRACTIVE == 'l2':
            #    l2_tensor = torch.tensor(l2_norms_before_norm)
            #    wandb.log({
            #        'lipschitz/conv_2_max': torch.max(l2_tensor).item(),
            #        'lipschitz/conv_2_mean': torch.mean(l2_tensor).item(),
            #        'lipschitz/conv_2_min': torch.min(l2_tensor).item()
            #    })

            wandb.log({'train/loss': loss.item(),
                       'train/acc@1': prec1[0],
                       'train/avg_loss': losses.avg,
                       'train/avg_acc@1': top1.avg,
                       'time/forward_ms': forward_time.avg,
                       'time/backward_ms': backward_time.avg})
            
        global_steps += 1
        writer_dict['train_global_steps'] = global_steps
        
        if factor > 0 and global_steps > config.TRAIN.PRETRAIN_STEPS and (deq_steps+1) % update_freq == 0:
             logger.info(f'Note: Adding 0.1 to Jacobian regularization weight.')

    epoch_duration = time.time() - epoch_start_time
    wandb.log({'train/1epoch_time': epoch_duration/60,
               'train/lr': optimizer.param_groups[0]['lr']})

def validate(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
             writer_dict=None, topk=(1,5), spectral_radius_mode=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    spectral_radius_mode = spectral_radius_mode and (epoch % 10 == 0)
    if spectral_radius_mode:
        sradiuses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    writer = writer_dict['writer'] if writer_dict else None

    forward_time = AverageMeter()
    backward_time = AverageMeter()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # switch to evaluate mode
    model.eval()

    start_time = time.time()

    with torch.no_grad():
        end = time.time()
        # tk0 = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
        for i, (input, target) in enumerate(val_loader):
            start_event.record()
            
            # compute output
            output, _, sradius = model(input, 
                                 train_step=(-1 if epoch < 0 else (lr_scheduler._step_count-1)),
                                 compute_jac_loss=False, spectral_radius_mode=spectral_radius_mode,
                                 writer=writer)
            target = target.cuda(non_blocking=True)
            loss = criterion(output, target)

            end_event.record()
            torch.cuda.synchronize()
            forward_time.update(start_event.elapsed_time(end_event))

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, topk=topk)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            if spectral_radius_mode:
                sradius = sradius.mean()
                sradiuses.update(sradius.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    if spectral_radius_mode:
        logger.info(f"Spectral radius over validation set: {sradiuses.avg}")    
    msg = 'Test: Time {batch_time.avg:.3f}\t' \
            'Loss {loss.avg:.4f}\t' \
            'Acc@1 {top1.avg:.3f}\t'.format(
                batch_time=batch_time, loss=losses, top1=top1)

    test_duration = time.time() - start_time
    wandb.log({'test/1epoch_time': test_duration/60,
               'test/loss': loss.item(),
               'test/acc@1': prec1[0],
               'test/avg_loss': losses.avg,
               'test/avg_acc@1': top1.avg,
               'time/test_forward_ms': forward_time.avg})
    
    if 5 in topk:
        msg += 'Acc@5 {top5.avg:.3f}\t'.format(top5=top5)
    logger.info(msg)

    if writer:
        writer.add_scalar('accuracy/valid_top1', top1.avg, epoch)
        if spectral_radius_mode:
            writer.add_scalar('stability/sradius', sradiuses.avg, epoch)

    return top1.avg
