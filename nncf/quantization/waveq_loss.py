import math
import torch

from nncf.compression_method_api import CompressionLoss
from nncf.quantization import layers

# передать сюда quant modules
# отадаю в функу что рисует график comp module
# график в тесте!
# отдаю сюда квант module
# опдаю веса в quant mod и получаю квант веса
# BaseQuantizer
# как из BaseQuantizer выжимать веса или
# как из compressed model выжимать веса
class WaveQLoss(CompressionLoss):

    def __init__(self, model):
        self.modules = model.children()

    def forward(self):
        loss = 0
        for layer in self.modules:
            # data extract from modules
            loss += WaveQLoss.waveq_loss_per_layer_sum(layer.weight.data)
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
