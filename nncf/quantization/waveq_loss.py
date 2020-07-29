import math
import torch

from nncf.compression_method_api import CompressionLoss
from nncf.quantization.algo import BaseQuantizer
from nncf.quantization.layers import SymmetricQuantizer, AsymmetricQuantizer


class LossHook:

    def __init__(self, quant_module: BaseQuantizer):
        self.input_tensor = None
        self.out_tensor = None
        self.quant_module = quant_module

    def fill_post_hook(self, module, inputs=None, outputs=None):
        self.input_tensor = inputs[0]
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
    def waveq_loss_for_tensor(tensor: torch.tensor, ratio=1, levels=16, input_low=0, input_range=1):
        # device problem
        # check device
        # check autograd
        return ratio * torch.square(torch.sin((tensor + input_low) / input_range
                                              * (levels - 1) * math.pi)) / levels

    @staticmethod
    def waveq_loss_per_hook_sum(hook: LossHook, ratio=1):
        # selection of quant params (bits, level_low, level_high, scale) from quant module
        # selection of quantizaton type and calculate level_low, level_high

        level_high, level_low, levels = get_quant_module_params(hook.quant_module)
        input_low, input_range = get_input_low_input_range(level_low=level_low
                                                           , level_high=level_high, scale=hook.quant_module.scale)
        out = WaveQLoss.waveq_loss_for_tensor(hook.input_tensor, ratio,
                                              levels=levels, input_low=input_low, input_range=input_range)
        return torch.sum(out)


def get_quant_module_params(quant_module: BaseQuantizer):
    try:
        quant_module.signed
        level_high, level_low, levels = quant_module.calculate_level_ranges(
            quant_module.num_bits, quant_module.signed, quant_module.is_weights)
    except:
        level_high, level_low, levels = quant_module.calculate_level_ranges(quant_module.num_bits)
    return level_high, level_low, levels


def get_input_low_input_range(level_low, level_high, scale):
    return scale * (level_low / level_high), scale - scale * (level_low / level_high)
