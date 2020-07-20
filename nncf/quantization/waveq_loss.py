import math
import torch

from nncf.compression_method_api import CompressionLoss
from nncf.quantization.layers import QUANTIZATION_MODULES
from nncf.utils import get_all_modules_by_type
from nncf.nncf_network import NNCFNetwork


class WaveQLoss(CompressionLoss):

    def __init__(self, model: NNCFNetwork):
        self.model = model
        # upper line doesn't work
        # "cannot assign module before Module.__init__() call")
        self.modules = model.children()
        self.quantize_modules = self.model_to_quantize_modules_convert(self.model)
        self.hook_handlers = None

    def model_to_quantize_modules_convert(self, model):
        for class_type in QUANTIZATION_MODULES.registry_dict.values():
            quantization_type = class_type.__name__
            module_dict = get_all_modules_by_type(self.model, quantization_type)
        return module_dict

    def calc_quant_tensor_by_module(self):
        self.hook_handlers = []
        for module in self.quantize_modules:
            hook_holder = LossHookHolder()
            self.hook_handlers.append(torch.nn.Module.register_forward_hook(hook_holder.calc_hook()))
        self.model.do_dummy_forward()

    def forward(self):
        loss = 0
        self.calc_quant_tensor_by_module()
        for hooker in self.hook_handlers:
            loss += WaveQLoss.waveq_loss_per_layer_sum(hooker.out_tensor)
        return loss

    def statistics(self):
        # printable stats
        pass

    def get_loss_stats_tensor(self, ratio=1, quant=8):  # list
        out = []
        for layer in self.modules:
            out.append(WaveQLoss.waveq_loss_per_layer_tensor(layer.weight.data, ratio=ratio, quant=quant))
        return out

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

    def calc_hook(self, module, outputs):
        self.out_tensor = outputs
