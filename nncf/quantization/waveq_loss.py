import math
import torch

from nncf.compression_method_api import CompressionLoss
from nncf.quantization.algo import BaseQuantizer


class LossHook:

    def __init__(self, quant_module: BaseQuantizer):
        self.input_tensor = None
        self.out_tensor = None
        self.quant_module = quant_module

    def fill_post_hook(self, module, inputs=None, outputs=None):
        self.input_tensor = inputs[0].data
        self.out_tensor = outputs


class WaveQLoss(CompressionLoss):

    def __init__(self, quantize_modules, ratio=1):
        super().__init__()
        self.ratio = ratio
        self.quantize_modules = quantize_modules
        self.post_hook_handlers = None
        self.pre_hook_handlers = None
        self.hooks = None
        self.set_up_hooks()

    def set_up_hooks(self):
        self.post_hook_handlers = []
        self.hooks = []
        for module in self.quantize_modules:
            hook = LossHook(module)
            self.post_hook_handlers.append(module.register_forward_hook(hook.fill_post_hook))
            self.hooks.append(hook)

    def forward(self):
        loss = 0
        for hooker in self.hooks:
            loss += WaveQLoss.waveq_loss_per_hook_sum(hooker, ratio=self.ratio)
        return loss

    @staticmethod
    def waveq_loss_for_tensor(tensor: torch.tensor, ratio=1, quant=8, scale=1):
        return ratio * torch.square(torch.sin(math.pi * tensor
                                              * (math.pow(2, quant) - 2) / float(scale) ) / (math.pow(2, quant)))

    @staticmethod
    def waveq_loss_per_hook_sum(hook: LossHook, ratio=1):
        # selection of quant params (bits, edges) from quant module
        out = WaveQLoss.waveq_loss_for_tensor(hook.input_tensor, ratio,
                                              quant=hook.quant_module.num_bits, scale=hook.quant_module.scale)
        return float(torch.sum(out))
