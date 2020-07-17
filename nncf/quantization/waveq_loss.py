import math
import torch

from nncf.compression_method_api import CompressionLoss


class WaveQLoss(CompressionLoss):

    def __init__(self, convlayers=None):
        self.convlayers = convlayers
        # how to implements quant bits? per layer

    def forward(self):
        loss = 0
        for layer in self.convlayers:
            loss += WaveQLoss.waveq_loss_per_layer(layer.weight.data)
        return loss

    def statistics(self):
        # printable stats
        pass

    @staticmethod
    def waveq_loss_per_layer(tensor: torch.tensor, ratio=1, quant=8):
        out = ratio * torch.square(torch.sin(math.pi * tensor
                                             * (math.pow(2, quant) - 1))) / math.pow(2, quant)
        return float(torch.sum(out))
