import math
import torch

from nncf.compression_method_api import CompressionLoss
from nncf.quantization.layers import BaseQuantizer, SymmetricQuantizer, AsymmetricQuantizer
from nncf.quantization.quantize_functions import TuneRange


class LossHook:

    def __init__(self, quant_module: BaseQuantizer):
        self.data = None
        self.out_tensor = None
        self.quant_module = quant_module

    def fill_post_hook(self, module, inputs=None, outputs=None):
        self.data = inputs[0]
        self.out_tensor = outputs


class WaveQLoss(CompressionLoss):
    # change to weights quant cnly
    # import loop
    def __init__(self, quantization_ctrl, ratio=1):
        super().__init__()
        self.ratio = ratio
        self.quantize_modules = list(quantization_ctrl.weight_quantizers.values())
        self.post_hook_handlers = None
        self.pre_hook_handlers = None
        self.hooks = None
        self.set_up_hooks()
        self.bottom_limit = None

    def set_up_hooks(self):
        self.post_hook_handlers = []
        self.hooks = []
        for module in self.quantize_modules:
            hook = LossHook(module)
            self.post_hook_handlers.append(module.register_forward_hook(hook.fill_post_hook))
            self.hooks.append(hook)

    def forward(self):
        loss = 0
        for hook_info in self.get_hook_data():
            loss += WaveQLoss.waveq_loss_per_hook_sum(hook_info, ratio=self.ratio)
        self.bottom_limit = loss / self.ratio
        return loss

    def get_hook_data(self):
        output = []
        for hook in self.hooks:
            info_dict = {}
            info_dict['data'] = hook.data
            info_dict['quant_module'] = hook.quant_module
            output.append(info_dict)
        return output

    def statistics(self): # dict
        return {'bottom_lim': self.bottom_limit}

    @staticmethod
    def waveq_loss_for_tensor(tensor: torch.tensor, ratio=1, levels=16, input_low=0, input_range=1):
        return ratio * torch.square(torch.sin((tensor + input_low) / input_range
                                              * (levels - 1) * math.pi)) / levels

    @staticmethod
    def waveq_loss_per_hook_sum(hook_info: dict, ratio=1):
        level_low, level_high, levels = hook_info['quant_module'].calculate_level_ranges(hook_info['quant_module'].num_bits,
                                                                                         hook_info['quant_module'].signed,
                                                                                         hook_info['quant_module'].is_weights)
        input_low, input_range = hook_info['quant_module'].calculate_inputs()
        out = WaveQLoss.waveq_loss_for_tensor(hook_info['data'], ratio,
                                              levels=levels, input_low=input_low, input_range=input_range)
        return torch.sum(out)
