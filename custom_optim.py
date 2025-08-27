import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional
import math

class SGD(Optimizer):
    """
    Custom SGD optimizer that supports standard gradient descent (gd) and exponential gradient (eg).
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, update_alg="gd",
                 freeze_gd_signs=False, freeze_gd_signs_th=1e-18):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if update_alg not in ["gd", "eg", "gd_sign"]:
            raise ValueError("Invalid update_alg value: {}".format(update_alg))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        update_alg=update_alg, freeze_gd_signs=freeze_gd_signs,
                        freeze_gd_signs_th=freeze_gd_signs_th)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                update_alg=group['update_alg'],
                freeze_gd_signs=group['freeze_gd_signs'],
                freeze_gd_signs_th=group['freeze_gd_signs_th'])

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        update_alg: str,
        freeze_gd_signs: bool,
        freeze_gd_signs_th: float
        ):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            if update_alg == "gd":
                d_p = d_p.add(param, alpha=weight_decay)
            elif update_alg == "eg":
                d_p = d_p.add(param.sign(), alpha=weight_decay)
        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        if update_alg == "gd" and freeze_gd_signs:
            s = param.sign()
            param.add_(d_p, alpha=-lr)
            flip_idx = (s * param) < 0
            param[flip_idx] = freeze_gd_signs_th * s[flip_idx]
        elif update_alg == 'gd':
            param.add_(d_p, alpha=-lr)
        elif update_alg == "eg":
            param.mul_(torch.exp(param.sign() * d_p * -lr))

class AdamWeg(Optimizer):
    """
    This is a slightly stripped down & modified version of torch.optim.AdamW from torch 1.13
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize, capturable=capturable)
        super(AdamWeg, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamWeg does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                state_steps.append(state['step'])
            adamw_eg(params_with_grad,
                     grads,
                     exp_avgs,
                     exp_avg_sqs,
                     max_exp_avg_sqs,
                     state_steps,
                     amsgrad=amsgrad,
                     beta1=beta1,
                     beta2=beta2,
                     lr=group['lr'],
                     weight_decay=group['weight_decay'],
                     eps=group['eps'],
                     maximize=group['maximize'],
                     foreach=group['foreach'],
                     capturable=group['capturable'])
        return loss

def adamw_eg(params: List[Tensor],
             grads: List[Tensor],
             exp_avgs: List[Tensor],
             exp_avg_sqs: List[Tensor],
             max_exp_avg_sqs: List[Tensor],
             state_steps: List[Tensor],
             foreach: bool = None,
             capturable: bool = False,
             *,
             amsgrad: bool,
             beta1: float,
             beta2: float,
             lr: float,
             weight_decay: float,
             eps: float,
             maximize: bool):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")
    if foreach is None:
        foreach = False
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw
    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_adamw(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         max_exp_avg_sqs: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         amsgrad: bool,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float,
                         eps: float,
                         maximize: bool,
                         capturable: bool):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)
        step_t += 1
        param.mul_(1 - lr * weight_decay)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if capturable:
            step = step_t
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)
            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()
            bias_correction2_sqrt = bias_correction2.sqrt()
            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            param.mul_(torch.exp(param.sign() * exp_avg / denom))
        else:
            step = step_t.item()
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)
            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            param.mul_(torch.exp(-step_size * param.sign() * exp_avg / denom))

def _multi_tensor_adamw(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool):
    if len(params) == 0:
        return
    if capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."
    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]
    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]
    exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avg_sqs]
    params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]
    torch._foreach_add_(state_steps, 1)
    torch._foreach_mul_(params, 1 - lr * weight_decay)
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)
    if capturable:
        bias_correction1 = [torch.pow(beta1, step) for step in state_steps]
        bias_correction2 = [torch.pow(beta2, step) for step in state_steps]
        torch._foreach_sub_(bias_correction1, 1)
        torch._foreach_sub_(bias_correction2, 1)
        torch._foreach_neg_(bias_correction1)
        torch._foreach_neg_(bias_correction2)
        step_size = torch._foreach_div(bias_correction1, lr)
        torch._foreach_reciprocal_(step_size)
        torch._foreach_neg_(step_size)
        bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)
        if amsgrad:
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(max_exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps_over_step_size)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)
        torch._foreach_mul_(params,
                            torch._foreach_exp(torch._foreach_div(
                                torch._foreach_mul(torch._foreach_sign(params),
                                                   exp_avgs),
                                denom)))
    else:
        bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
        bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]
        step_size = [(lr / bc) * -1 for bc in bias_correction1]
        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]
        if amsgrad:
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps)
        torch._foreach_mul_(params,
                            torch._foreach_exp(torch._foreach_div(
                                torch._foreach_mul(torch._foreach_mul(torch._foreach_sign(params),
                                                                      step_size),
                                                   exp_avgs),
                                denom)))

class AdamGD(Optimizer):
    """
    AdamGD optimizer - combines Adam's adaptive learning rate with standard gradient descent updates.
    This is similar to AdamWeg but uses additive updates instead of multiplicative exponential updates.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize, capturable=capturable)
        super(AdamGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamGD does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                state_steps.append(state['step'])
            adamw_gd(params_with_grad,
                     grads,
                     exp_avgs,
                     exp_avg_sqs,
                     max_exp_avg_sqs,
                     state_steps,
                     amsgrad=amsgrad,
                     beta1=beta1,
                     beta2=beta2,
                     lr=group['lr'],
                     weight_decay=group['weight_decay'],
                     eps=group['eps'],
                     maximize=group['maximize'],
                     foreach=group['foreach'],
                     capturable=group['capturable'])
        return loss

def adamw_gd(params: List[Tensor],
             grads: List[Tensor],
             exp_avgs: List[Tensor],
             exp_avg_sqs: List[Tensor],
             max_exp_avg_sqs: List[Tensor],
             state_steps: List[Tensor],
             foreach: bool = None,
             capturable: bool = False,
             *,
             amsgrad: bool,
             beta1: float,
             beta2: float,
             lr: float,
             weight_decay: float,
             eps: float,
             maximize: bool):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")
    if foreach is None:
        foreach = False
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw_gd
    else:
        func = _single_tensor_adamw_gd
    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_adamw_gd(params: List[Tensor],
                            grads: List[Tensor],
                            exp_avgs: List[Tensor],
                            exp_avg_sqs: List[Tensor],
                            max_exp_avg_sqs: List[Tensor],
                            state_steps: List[Tensor],
                            *,
                            amsgrad: bool,
                            beta1: float,
                            beta2: float,
                            lr: float,
                            weight_decay: float,
                            eps: float,
                            maximize: bool,
                            capturable: bool):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)
        step_t += 1
        
        # Apply weight decay
        if weight_decay != 0:
            param.add_(param, alpha=-weight_decay * lr)
        
        # Exponential moving average of gradient values
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Exponential moving average of squared gradient values
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        if capturable:
            step = step_t
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)
            step_size = lr / bias_correction1
            bias_correction2_sqrt = bias_correction2.sqrt()
            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            # Standard gradient descent update (additive)
            param.add_(exp_avg / denom, alpha=-step_size)
        else:
            step = step_t.item()
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)
            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            # Standard gradient descent update (additive)
            param.add_(exp_avg / denom, alpha=-step_size)

def _multi_tensor_adamw_gd(params: List[Tensor],
                           grads: List[Tensor],
                           exp_avgs: List[Tensor],
                           exp_avg_sqs: List[Tensor],
                           max_exp_avg_sqs: List[Tensor],
                           state_steps: List[Tensor],
                           *,
                           amsgrad: bool,
                           beta1: float,
                           beta2: float,
                           lr: float,
                           weight_decay: float,
                           eps: float,
                           maximize: bool,
                           capturable: bool):
    if len(params) == 0:
        return
    if capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."
    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]
    
    # Handle complex parameters
    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]
    exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avg_sqs]
    params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]
    
    # Update steps
    torch._foreach_add_(state_steps, 1)
    
    # Apply weight decay
    if weight_decay != 0:
        torch._foreach_add_(params, params, alpha=-weight_decay * lr)
    
    # Update biased first moment estimate
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
    
    # Update biased second raw moment estimate
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)
    
    if capturable:
        bias_correction1 = [torch.pow(beta1, step) for step in state_steps]
        bias_correction2 = [torch.pow(beta2, step) for step in state_steps]
        torch._foreach_sub_(bias_correction1, 1)
        torch._foreach_sub_(bias_correction2, 1)
        torch._foreach_neg_(bias_correction1)
        torch._foreach_neg_(bias_correction2)
        
        step_size = torch._foreach_div(bias_correction1, lr)
        torch._foreach_reciprocal_(step_size)
        torch._foreach_neg_(step_size)
        
        bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)
        
        if amsgrad:
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
            denom = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(denom, bias_correction2_sqrt)
            torch._foreach_add_(denom, eps)
        else:
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denom, bias_correction2_sqrt)
            torch._foreach_add_(denom, eps)
        
        # Standard gradient descent update (additive)
        torch._foreach_add_(params, torch._foreach_div(exp_avgs, denom), alpha=step_size)
    else:
        bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
        bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]
        step_size = [-lr / bc for bc in bias_correction1]
        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]
        
        if amsgrad:
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
            denom = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(denom, bias_correction2_sqrt)
            torch._foreach_add_(denom, eps)
        else:
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denom, bias_correction2_sqrt)
            torch._foreach_add_(denom, eps)
        
        # Standard gradient descent update (additive)
        torch._foreach_add_(params, torch._foreach_div(
            torch._foreach_mul(exp_avgs, step_size), denom))

class LNS_Madam(Optimizer):
    #LNS_Madam optimizer
    def __init__(self, params, lr=1/128, p_scale=3.0, g_bound=10.0, wd=None, momentum=0):
        self.p_scale = p_scale
        self.g_bound = g_bound
        self.wd = wd
        self.momentum = momentum
        self.dampening = 0
        defaults = dict(lr=lr)
        super(LNS_Madam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                d_p = p.grad.data

                if len(state) == 0:
                    state['max'] = self.p_scale*(p*p).mean().sqrt().item()
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                bias_correction = 1 - 0.999 ** state['step']
                state['exp_avg_sq'] = 0.999 * state['exp_avg_sq'] + 0.001 * p.grad.data**2
                
                g_normed = d_p / (state['exp_avg_sq']/bias_correction).sqrt()
                g_normed[torch.isnan(g_normed)] = 0
                g_normed.clamp_(-self.g_bound, self.g_bound)
                g_normed.round_() #rounded 
                
                if self.wd is not None: 
                    p.data *= torch.exp( -group['lr']*g_normed*torch.sign(p.data) - group['lr']*self.wd )
                else:
                    #p.data *= torch.exp( -group['lr']*g_normed*torch.sign(p.data) )
                    p.data *= 2.0**( -group['lr']*g_normed*torch.sign(p.data))
                p.data.clamp_(-state['max'], state['max'])

        return loss 