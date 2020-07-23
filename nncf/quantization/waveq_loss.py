import math
import torch

from nncf.compression_method_api import CompressionLoss


class WaveQLoss(CompressionLoss):

    def __init__(self, quantize_modules, ratio=1):
        super().__init__()
        self.ratio = ratio
        self.quantize_modules = quantize_modules
        self.hook_handlers = None
        self.hook_collectors = None
        self.set_up_hooks()

    def set_up_hooks(self):
        self.hook_handlers = []
        self.hook_collectors = []
        for module in self.quantize_modules:
            hook = LossHook(module)
            self.hook_handlers.append(module.register_forward_hook(hook.calc_hook))
            self.hook_collectors.append(hook)

    def forward(self):
        loss = 0
        for hooker in self.hook_collectors:
            loss += WaveQLoss.waveq_loss_per_layer_sum(hooker.out_tensor, ratio=self.ratio)
        return loss

    def calc_hook_waveq(self):
        loss = 0
        for hook_handler in self.hook_collectors:
            loss += WaveQLoss.waveq_loss_per_layer_sum(hook_handler.out_tensor, )
        return loss

    @staticmethod
    def waveq_loss_per_layer_sum(tensor: torch.tensor, ratio=1, quant=8):
        out = ratio * torch.square(torch.sin(math.pi * tensor
                                       * (math.pow(2, quant) - 1))) / math.pow(2, quant)
        return float(torch.sum(out))


class LossHook:

    def __init__(self, quant_module):
        self.out_tensor = None
        self.quant_module = quant_module

    def calc_hook(self, module, inputs, outputs):
        self.out_tensor = outputs
