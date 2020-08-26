import math
import torch


from nncf.compression_method_api import CompressionLoss, CompressionScheduler
from nncf.quantization.layers import BaseQuantizer, SymmetricQuantizer, AsymmetricQuantizer
from nncf.quantization.quantize_functions import TuneRange
from nncf.quantization.init_precision import PerturbationObserver
from nncf.quantization.algo import CompressionAlgorithmController


class LossHook:

    def __init__(self, quant_module: BaseQuantizer):
        self.data = None
        self.out_tensor = None
        self.quant_module = quant_module

    def fill_post_hook(self, module, inputs=None, outputs=None):
        self.data = inputs[0]
        self.out_tensor = outputs


class WaveQLoss(CompressionLoss):

    def __init__(self, quantization_ctrl, ratio=1):
        super().__init__()
        self.ratio = ratio
        self.quantize_modules = list(quantization_ctrl.weight_quantizers.values())
        self.post_hook_handlers = None
        self.pre_hook_handlers = None
        self.hooks = None
        self.set_up_hooks()
        self.bottom_limit = 0
        self.perturbation = 0

    def set_up_hooks(self):
        self.perturbation_observers_list = []
        self.post_hook_handlers = []
        self.hooks = []
        for module in self.quantize_modules:
            hook = LossHook(module)
            perturbation_observer = PerturbationObserver(None)
            module.register_forward_hook(perturbation_observer.calc_perturbation)
            self.post_hook_handlers.append(module.register_forward_hook(hook.fill_post_hook))
            self.hooks.append(hook)
            self.perturbation_observers_list.append(perturbation_observer)

    def forward(self):
        loss = 0
        for hook_info in self.get_hook_data():
            loss += WaveQLoss.waveq_loss_per_hook_sum(hook_info, ratio=self.ratio)
        self.perturbation = self.pert_calc()
        self.bottom_limit = loss / self.ratio
        return loss

    def pert_calc(self):
        perturbation = 0
        for pert_observer in self.perturbation_observers_list:
            perturbation += pert_observer.perturbation
        return perturbation

    def get_hook_data(self):
        output = []
        for hook in self.hooks:
            info_dict = {}
            info_dict['data'] = hook.data
            info_dict['quant_module'] = hook.quant_module
            output.append(info_dict)
        return output

    def statistics(self):  # dict
        return {'Bottom_Lim': float(self.bottom_limit), 'Quant_Perturbation': float(self.perturbation)}

    @staticmethod
    def waveq_loss_for_tensor(tensor: torch.tensor, ratio=1, levels=16, input_low=0, input_range=1):
        return ratio * torch.square(torch.sin((tensor + input_low) / input_range
                                              * (levels - 1) * math.pi)) / levels

    @staticmethod
    def waveq_loss_per_hook_sum(hook_info: dict, ratio=1):
        level_low, level_high, levels = hook_info['quant_module'].calculate_level_ranges(
            hook_info['quant_module'].num_bits,
            hook_info['quant_module'].signed,
            hook_info['quant_module'].is_weights)
        input_low, input_range = hook_info['quant_module'].calculate_inputs()
        out = WaveQLoss.waveq_loss_for_tensor(hook_info['data'], ratio,
                                              levels=levels, input_low=input_low, input_range=input_range)
        return torch.sum(out)


class WaveQScheduler(CompressionScheduler):

    def __init__(self, compression_ctrl: CompressionAlgorithmController):
        super().__init__()
        self.compression_ctrl = compression_ctrl

    def epoch_step(self, last=None):
        if last is None:
            last = self.last_epoch + 1
        self.last_epoch = last

    def _lambda_change(self):
        raise NotImplementedError


class WaveQEpochStepScheduler(WaveQScheduler):

    def __init__(self, compression_ctrl: CompressionAlgorithmController, epoch_steps: list):
        super(WaveQEpochStepScheduler, self).__init__(compression_ctrl)
        self.epoch_steps = epoch_steps

    def epoch_step(self, last=None):
        super().epoch_step()
        if self.last_epoch in self.epoch_steps:
            self._lambda_change()

    def _lambda_change(self):
        self.compression_ctrl.loss.ratio = self.compression_ctrl.loss.ratio * 10
