import torch
from torch.autograd import Function
from lib.layer_utils import list2vec, vec2list

class JFB_Function(Function):
    @staticmethod
    def forward(ctx, z_star, x_list, fullstage, *fullstage_params):
        ctx.x_list = x_list
        ctx.fullstage = fullstage
        ctx.fullstage_params = fullstage_params
        ctx.save_for_backward(z_star)
        
        ctx.cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in x_list]

        return z_star

    @staticmethod
    def backward(ctx, grad_z_star):
        z_star, = ctx.saved_tensors
        x_list = ctx.x_list
        fullstage = ctx.fullstage
        params = ctx.fullstage_params
        cutoffs = ctx.cutoffs
        
        with torch.enable_grad():
            z_star_attached = z_star.detach().requires_grad_()
            
            z_star_list = vec2list(z_star_attached, cutoffs)
            f_z_star_list = fullstage(z_star_list, x_list)

        param_grads = torch.autograd.grad(
            outputs=f_z_star_list,
            inputs=params,
            grad_outputs=vec2list(grad_z_star, cutoffs)
        )

        # z_star, x_list, fullstage, *fullstage_params
        return grad_z_star, None, None, *param_grads
