import math
import torch

from nncf.compression_method_api import CompressionLoss
from nncf.quantization.algo import BaseQuantizer


class WaveQLoss(CompressionLoss):

    def __init__(self, quantize_modules, ratio=1):
        super().__init__()
        self.ratio = ratio
        self.quantize_modules = quantize_modules
        self.post_hook_handlers = None
        self.post_hooks = None
        self.set_up_hooks()

    def set_up_hooks(self):
        self.post_hook_handlers = []
        self.post_hooks = []
        for module in self.quantize_modules:
            hook = LossHook(module)
            self.post_hook_handlers.append(module.register_forward_hook(hook.calc_hook))
            self.post_hooks.append(hook)

    def forward(self):
        loss = 0
        for hooker in self.post_hooks:
            loss += WaveQLoss.waveq_loss_per_layer_sum(hooker.out_tensor, ratio=self.ratio)
        return loss

    @staticmethod
    def waveq_loss_for_tensor(tensor: torch.tensor, ratio=1, quant=4):
        return ratio * torch.square(torch.sin(math.pi * tensor
                                              * (math.pow(2, quant) - 1))) / math.pow(2, quant)

    @staticmethod
    def waveq_loss_per_layer_sum(tensor: torch.tensor, ratio=1, quant=4):
        out = WaveQLoss.waveq_loss_for_tensor(tensor, ratio, quant)
        return float(torch.sum(out))


class LossHook:

    def __init__(self, quant_module: BaseQuantizer):
        # basequantizer
        self.out_tensor = None
        self.quant_module = quant_module

    def calc_hook(self, module, inputs, outputs):
        self.out_tensor = outputs
