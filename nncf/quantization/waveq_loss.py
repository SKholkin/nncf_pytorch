import math
import torch

from nncf.compression_method_api import CompressionLoss


class WaveQLoss(CompressionLoss):

    def __init__(self, quantize_modules):
        self.quantize_modules = quantize_modules
        self.hook_handlers = None

    def set_up_hooks(self):
        self.hook_handlers = []
        for module in self.quantize_modules:
            hook_holder = LossHookHolder()
            self.hook_handlers.append(module.register_forward_hook(hook_holder.calc_hook))

    def forward(self):
        # do i need forward at all?
        # where to calculate loss?
        loss = 0
        self.set_up_hooks()
        for hooker in self.hook_handlers:
            loss += WaveQLoss.waveq_loss_per_layer_sum(hooker.out_tensor)
        return loss

    def calc_hook_waveq(self):
        loss = 0
        for hook_handler in self.hook_handlers:
            loss += WaveQLoss.waveq_loss_per_layer_sum(hook_handler.out_tensor)
        return loss

    # remove
    def get_loss_stats_tensor(self, ratio=1, quant=8):  # list
        out = []
        for layer in self.modules:
            out.append(WaveQLoss.waveq_loss_per_layer_tensor(layer.weight.data, ratio=ratio, quant=quant))
        return out

    # probably remove
    @staticmethod
    def waveq_loss_per_layer_tensor(tensor: torch.tensor, ratio=1, quant=8):
        return ratio * torch.square(torch.sin(math.pi * tensor
                                              * (math.pow(2, quant) - 1))) / math.pow(2, quant)

    @staticmethod
    def waveq_loss_per_layer_sum(tensor: torch.tensor, ratio=1, quant=8):
        out = WaveQLoss.waveq_loss_per_layer_tensor(tensor, ratio=ratio, quant=quant)
        return float(torch.sum(out))


class LossHookHolder:

    def __init__(self):
        self.out_tensor = None

    def calc_hook(self, module, inputs, outputs):
        self.out_tensor = outputs
