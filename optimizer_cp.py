from copy import deepcopy
from typing import Union, Tuple

import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    def __init__(
            self,
            named_params,
            lr: Union[float, torch.Tensor] = 1e-3,
            scaling_factor: float = 2.,
            weight_decay: float = 0,
            momentum=0.,
            interval=100,
            optimizer_states="reset",
            norm_limit=True,
            args=None,
    ):
        names = []
        params = []
        for n, p in named_params:
            names.append(n)
            params.append(p)

        self.interval = interval
        self.optimizer_states = optimizer_states
        self.norm_limit = norm_limit
        self.args = args

        defaults = dict(
            lr=lr,
            names=names,
            scaling_factor=scaling_factor,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        g = torch.Generator(self.args.device)
        for group in self.param_groups:
            scaling_factor = group["scaling_factor"]
            momentum = group['momentum']
            param_list = []
            param_dict = dict(zip(group["names"], group["params"]))
            for n, p in param_dict.items():

                if 'lora' in n:
                    param_list.append(p)
                    if len(param_list) == 2:
                        base_name = n[: n.find('lora')]
                        name = base_name + "base_layer.weight"
                        size = param_dict[n].shape
                    else:
                        continue
                elif p.grad is None:
                    continue
                else:
                    name = n
                    size = p.shape

                state = self.state[name]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0)
                    if momentum != 0:
                        state["momentum"] = torch.zeros(size).to(p.device).to(p.dtype)

                state['step'] += 1

                # step
                if len(param_list) == 2:
                    grad = param_list[1].grad / scaling_factor
                else:
                    grad = p.grad

                if momentum != 0:
                    buf = state["momentum"]
                    buf.mul_(momentum).add_(grad, alpha=1)
                    grad = buf

                step_size = group['lr']

                if len(param_list) != 2:
                    p.mul_(1 - group["weight_decay"] * group["lr"])
                    p.add_(grad, alpha=-step_size)
                else:
                    param_dict[name].mul_(1 - group["weight_decay"] * group["lr"])
                    grad = grad @ param_list[0]
                    param_dict[name].add_(
                        grad,
                        alpha=-step_size
                    )
                    if state['step'] % self.interval == 0:
                        old_param = deepcopy(param_list[0])
                        torch.nn.init.normal_(param_list[0], mean=0, std=1.0 / param_list[0].shape[0] ** 0.5,
                                              generator=g)
                        if 'momentum' in state:
                            if self.optimizer_states == 'reset':
                                torch.nn.init.zeros_(state['momentum'])
                            elif self.optimizer_states == 'transform':
                                state['momentum'] = state['momentum'] @ old_param @ param_list[0].T * (
                                            param_list[0].shape[0] / param_list[0].shape[1])
                            elif self.optimizer_states == "unchanged":
                                pass
                    param_list = []
        return loss


class AdamW(Optimizer):
    def __init__(
            self,
            named_params,
            lr: Union[float, torch.Tensor] = 1e-3,
            scaling_factor: float = 2.,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            interval=100,
            optimizer_states="reset",
            args=None
    ):
        names = []
        params = []
        for n, p in named_params:
            names.append(n)
            params.append(p)

        self.interval = interval
        self.optimizer_states = optimizer_states
        self.args = args
        defaults = dict(
            lr=lr,
            names=names,
            scaling_factor=scaling_factor,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            scaling_factor = group["scaling_factor"]

            g = torch.Generator(self.args.device)
            param_list = []
            param_dict = dict(zip(group["names"], group["params"]))
            for n, p in param_dict.items():
                if 'lora' in n:
                    if 'lora_B' in n:
                        size = param_dict[n].shape
                    param_list.append(p)
                    if len(param_list) == 2:
                        base_name = n[: n.find('lora')]
                        name = base_name + "base_layer.weight"
                    else:
                        continue
                elif p.grad is None:
                    continue
                else:
                    name = n
                    size = p.shape

                state = self.state[name]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros(size).to(p.device).to(p.dtype)

                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros(size).to(p.device).to(p.dtype)

                if len(param_list) == 2:
                    grad = param_list[1].grad / scaling_factor
                else:
                    grad = p.grad

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group['lr']
                denom = (exp_avg_sq.sqrt() / bias_correction2 ** 0.5).add_(group['eps'])

                if len(param_list) != 2:
                    p.mul_(1 - group["weight_decay"] * group["lr"])
                    p.addcdiv_(exp_avg / bias_correction1, denom, value=-step_size)
                else:
                    param_dict[name].mul_(1 - group["weight_decay"] * group["lr"])
                    param_dict[name].add_(
                            (exp_avg / bias_correction1 / denom).to(param_list[0].dtype) @ param_list[0],
                            alpha=-step_size
                        )
                    if state['step'] % self.interval == 0:
                        old_param = deepcopy(param_list[0])
                        torch.nn.init.normal_(param_list[0], mean=0, std=1.0 / param_list[0].shape[0] ** 0.5,
                                              generator=g)
                        if 'exp_avg' in state:
                            if self.optimizer_states == 'reset':
                                torch.nn.init.zeros_(state['exp_avg'])
                                torch.nn.init.zeros_(state['exp_avg_sq'])
                            elif self.optimizer_states == 'transform':
                                state['exp_avg'] = state['exp_avg'] @ old_param @ param_list[0].T * (
                                            param_list[0].shape[0] / param_list[0].shape[1])
                                # state['exp_avg_sq'] = state['exp_avg_sq'] @ old_param @ param_list[0].T
                            elif self.optimizer_states == "unchanged":
                                pass
                    param_list = []

        return loss


class RandomizedSGD(Optimizer):
    def __init__(
            self,
            named_params,
            lr: Union[float, torch.Tensor] = 1e-3,
            momentum=0.9,
            weight_decay: float = 0,
            proj_type='left',
            interval=100,
            args=None,
    ):
        names = []
        params = []
        for n, p in named_params:
            names.append(n)
            params.append(p)

        self.args = args
        self.proj_type = proj_type
        self.interval = interval
        defaults = dict(
            lr=lr,
            names=names,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            grad_dict = {}
            for group in self.param_groups:
                param_dict = dict(zip(group["names"], group["params"]))
                for name, param in param_dict.items():
                    if 'lora_A' in name:
                        torch.nn.init.zeros_(param)
                        param.requires_grad = True
                    if 'lora_B' in name:
                        g = param.grad
                        q, r = torch.qr(g / (self.args.lora_alpha / self.args.lora_r))
                        grad_dict[name.replace('lora_B', 'lora_A')] = q
                        param.grad = None
                        param.requires_grad = False
                        param.copy_(q)

            with torch.enable_grad():
                loss = closure()
                loss.backward()

            for group in self.param_groups:
                param_dict = dict(zip(group["names"], group["params"]))
                for name, param in param_dict.items():
                    if 'lora_A' in name:
                        g = param.grad
                        u, s, v = torch.svd(g / (self.args.lora_alpha / self.args.lora_r))
                        weight_name = name[: name.find('lora')] + "base_layer.weight"
                        grad_dict[weight_name] = (grad_dict[name] @ u, s, v)
                        del grad_dict[name]
                        param.grad = None
                        param.requires_grad = False
                        torch.nn.init.normal_(param, mean=0, std=1)

                    if 'lora_B' in name:
                        torch.nn.init.zeros_(param)
                        param.requires_grad = True

        for group in self.param_groups:
            momentum = group['momentum']
            param_dict = dict(zip(group["names"], group["params"]))
            for name, p in param_dict.items():
                if p.grad is None and name not in grad_dict:
                    continue

                if p.grad is not None:
                    grad = p.grad
                    size = p.shape
                else:
                    u, s, v = grad_dict[name]
                    grad = u @ torch.diag(s) @ v.T
                    if self.proj_type == 'left':
                        size = v.T.shape
                    else:
                        size = u.shape

                state = self.state[name]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["momentum"] = torch.zeros(size).to(p.device).to(p.dtype)
                    if self.proj_type == 'left':
                        state["project_matrix"] = torch.zeros_like(u.T)
                    else:
                        state["project_matrix"] = torch.zeros_like(v)

                if name in grad_dict:
                    if self.proj_type == 'left':
                        if state["step"] % self.interval == 0:
                            state["project_matrix"] = u.T
                        grad = state["project_matrix"] @ grad
                    else:
                        if state["step"] % self.interval == 0:
                            state["project_matrix"] = v
                        grad = grad @ state["project_matrix"]

                buf = state["momentum"]
                buf.mul_(momentum).add_(grad, alpha=1)
                grad = buf
                if name in grad_dict:
                    grad = state["project_matrix"].T @ grad if self.proj_type == 'left' else grad @ \
                                                                                                    state[
                                                                                                        "project_matrix"].T
                step_size = group['lr']

                p.mul_(1 - group["weight_decay"] * group["lr"])
                p.add_(grad, alpha=-step_size)

        return loss


class RandomizedApollo(Optimizer):
    def __init__(
            self,
            named_params,
            lr: Union[float, torch.Tensor] = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            proj_type='left',
            norm_limit=True,
            interval=100,
            args=None,
    ):
        names = []
        params = []
        for n, p in named_params:
            names.append(n)
            params.append(p)

        self.proj_type = proj_type
        self.norm_limit = norm_limit
        self.interval = interval
        self.args = args
        defaults = dict(
            lr=lr,
            names=names,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            grad_dict = {}
            for group in self.param_groups:
                param_dict = dict(zip(group["names"], group["params"]))
                for name, param in param_dict.items():
                    if 'lora_A' in name:
                        torch.nn.init.zeros_(param)
                        param.requires_grad = True
                    if 'lora_B' in name:
                        g = param.grad
                        q, r = torch.qr(g / (self.args.lora_alpha / self.args.lora_r))
                        grad_dict[name.replace('lora_B', 'lora_A')] = q
                        param.grad = None
                        param.requires_grad = False
                        param.copy_(q)

            with torch.enable_grad():
                loss = closure()
                loss.backward()

            for group in self.param_groups:
                param_dict = dict(zip(group["names"], group["params"]))
                for name, param in param_dict.items():
                    if 'lora_A' in name:
                        g = param.grad
                        u, s, v = torch.svd(g / (self.args.lora_alpha / self.args.lora_r))
                        weight_name = name[: name.find('lora')] + "base_layer.weight"
                        grad_dict[weight_name] = (grad_dict[name] @ u, s, v)
                        del grad_dict[name]
                        param.grad = None
                        param.requires_grad = False
                        torch.nn.init.normal_(param, mean=0, std=1)

                    if 'lora_B' in name:
                        torch.nn.init.zeros_(param)
                        param.requires_grad = True

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            param_dict = dict(zip(group["names"], group["params"]))
            for name, p in param_dict.items():
                if p.grad is None and name not in grad_dict:
                    continue

                if p.grad is not None:
                    grad = p.grad
                    size = p.shape
                else:
                    u, s, v = grad_dict[name]
                    original_grad = u @ torch.diag(s) @ v.T
                    if self.proj_type == 'left':
                        size = v.T.shape
                    else:
                        size = u.shape

                state = self.state[name]

                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros(size).to(p.device).to(p.dtype)
                    state["exp_avg_sq"] = torch.zeros(size).to(p.device).to(p.dtype)

                if name in grad_dict:
                    if self.proj_type == 'left':
                        if state["step"] % self.interval == 0:
                            state["project_matrix"] = u.T
                        grad = state["project_matrix"] @ original_grad
                    else:
                        if state["step"] % self.interval == 0:
                            state["project_matrix"] = v
                        grad = original_grad @ state["project_matrix"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group['lr'] * bias_correction2 ** 0.5 / bias_correction1

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                if p.grad is not None:
                    p.mul_(1 - group["weight_decay"] * group["lr"])
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    param_dict[name].mul_(1 - group["weight_decay"] * group["lr"])

                    norm_dim = 0 if self.proj_type == 'left' else 1
                    grad_scale = (
                            torch.norm(exp_avg / denom, dim=norm_dim) /
                            (torch.norm(grad, dim=norm_dim) + 1e-8)
                    )

                    scaled_grad = original_grad @ torch.diag(grad_scale) if self.proj_type == 'left' else torch.diag(
                        grad_scale) @ original_grad

                    if self.norm_limit:
                        if "scaled_grad" in state:
                            scaled_grad_norm = torch.norm(scaled_grad)
                            limiter = max(
                                scaled_grad_norm /
                                (state["scaled_grad"] + 1e-8),
                                1.01,
                            ) / 1.01
                            scaled_grad = scaled_grad / limiter
                            state["scaled_grad"] = scaled_grad_norm / limiter
                        else:
                            state["scaled_grad"] = torch.norm(scaled_grad)

                    param_dict[name].add_(scaled_grad,
                                          alpha=-step_size)

        return loss


class RandomizedGalore(Optimizer):
    def __init__(
            self,
            named_params,
            lr: Union[float, torch.Tensor] = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            proj_type='left',
            interval=100,
            args=None
    ):
        names = []
        params = []
        for n, p in named_params:
            names.append(n)
            params.append(p)

        self.proj_type = proj_type
        self.interval = interval
        self.args=args
        defaults = dict(
            lr=lr,
            names=names,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            grad_dict = {}
            for group in self.param_groups:
                param_dict = dict(zip(group["names"], group["params"]))
                for name, param in param_dict.items():
                    if 'lora_A' in name:
                        torch.nn.init.zeros_(param)
                        param.requires_grad = True
                    if 'lora_B' in name:
                        g = param.grad
                        q, r = torch.qr(g / (self.args.lora_alpha / self.args.lora_r))
                        grad_dict[name.replace('lora_B', 'lora_A')] = q
                        param.grad = None
                        param.requires_grad = False
                        param.copy_(q)

            with torch.enable_grad():
                loss = closure()
                loss.backward()

            for group in self.param_groups:
                param_dict = dict(zip(group["names"], group["params"]))
                for name, param in param_dict.items():
                    if 'lora_A' in name:
                        g = param.grad
                        u, s, v = torch.svd(g / (self.args.lora_alpha / self.args.lora_r))
                        weight_name = name[: name.find('lora')] + "base_layer.weight"
                        grad_dict[weight_name] = (grad_dict[name] @ u, s, v)
                        del grad_dict[name]
                        param.grad = None
                        param.requires_grad = False
                        torch.nn.init.normal_(param, mean=0, std=1)

                    if 'lora_B' in name:
                        torch.nn.init.zeros_(param)
                        param.requires_grad = True

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            param_dict = dict(zip(group["names"], group["params"]))
            for name, p in param_dict.items():
                if p.grad is None and name not in grad_dict:
                    continue

                if p.grad is not None:
                    grad = p.grad
                    size = p.shape
                else:
                    u, s, v = grad_dict[name]
                    grad = u @ torch.diag(s) @ v.T
                    if self.proj_type == 'left':
                        size = v.T.shape
                    else:
                        size = u.shape

                state = self.state[name]

                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros(size).to(p.device).to(p.dtype)
                    state["exp_avg_sq"] = torch.zeros(size).to(p.device).to(p.dtype)

                if name in grad_dict:
                    if self.proj_type == 'left':
                        if state["step"] % self.interval == 0:
                            state["project_matrix"] = u.T
                        grad = state["project_matrix"] @ grad
                    else:
                        if state["step"] % self.interval == 0:
                            state["project_matrix"] = v
                        grad = grad @ state["project_matrix"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group['lr'] * bias_correction2 ** 0.5 / bias_correction1

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                if p.grad is not None:
                    p.mul_(1 - group["weight_decay"] * group["lr"])
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    param_dict[name].mul_(1 - group["weight_decay"] * group["lr"])

                    grad = state["project_matrix"].T @ grad if self.proj_type == 'left' else grad @ state["project_matrix"].T
                    param_dict[name].add_(grad, alpha=-step_size)

        return loss


class Flora(Optimizer):
    def __init__(
            self,
            params,
            lr: Union[float, torch.Tensor] = 1e-3,
            scaling_factor: float = 2.,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            interval=100,
            optimizer_states="reset",
            args=None
    ):

        self.interval = interval
        self.optimizer_states = optimizer_states
        self.args = args
        defaults = dict(
            lr=lr,
            scaling_factor=scaling_factor,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            param_dict = dict(zip(group["names"], group["params"]))
            for n, p in param_dict.items():
                if p.grad is None:
                    continue

                compress = True if "rank" in group else False
                state = self.state[n]

                if compress:
                    proj = state.get("proj", torch.randn(size=(p.grad.size(1), self.args.lora_r), device=p.grad.device))
                    grad = p.grad @ proj
                else:
                    grad = p.grad

                size = grad.shape

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros(size).to(p.device).to(p.dtype)

                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros(size).to(p.device).to(p.dtype)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group['lr']
                denom = (exp_avg_sq.sqrt() / bias_correction2 ** 0.5).add_(group['eps'])

                p.mul_(1 - group["weight_decay"] * group["lr"])
                if not compress:
                    p.addcdiv_(exp_avg / bias_correction1, denom, value=-step_size)
                else:
                    p.add_(
                            (exp_avg / bias_correction1 / denom).to(grad.dtype) @ proj.T,
                            alpha=-step_size
                        )
                    if state['step'] % self.interval == 0:
                        old_param = deepcopy(proj)
                        torch.nn.init.normal_(proj, mean=0, std=1.0 / proj.shape[0] ** 0.5)
                        if 'exp_avg' in state:
                            if self.optimizer_states == 'reset':
                                torch.nn.init.zeros_(state['exp_avg'])
                                torch.nn.init.zeros_(state['exp_avg_sq'])
                            elif self.optimizer_states == 'transform':
                                state['exp_avg'] = state['exp_avg'] @ old_param @ proj.T * (
                                            proj.shape[0] / proj.shape[1])
                                # state['exp_avg_sq'] = state['exp_avg_sq'] @ old_param @ param_list[0].T
                            elif self.optimizer_states == "unchanged":
                                pass
        return loss
