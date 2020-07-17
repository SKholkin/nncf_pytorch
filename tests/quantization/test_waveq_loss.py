import pytest
from nncf.quantization.waveq_loss import WaveQLoss
import torch
import math


def test_waveq_loss_0():
    layers = list()
    layers.append(torch.nn.Conv2d(3, 2, 3, 3))
    layers[0].weight.data = torch.tensor([[[[0.002], [0.0098]],
                               [[0.0025], [0.0021]]],
                              [[[0.0140], [0.098]],
                               [[0.033], [0.0166]]],
                              [[[0.0206], [0.0284]],
                               [[0.04], [0.029]]],
                              [[[0.0145], [0.0137]],
                               [[0.003], [0.0294]]]], dtype=torch.float32)
    layers.append(torch.nn.Conv2d(1, 1, 2, 2))
    layers[1].weight.data = torch.tensor([[[[0.033], [0.0137]], [[0.0206], [0.0166]]]])
    loss_module = WaveQLoss(convlayers=layers)
    predicted_loss = 0.05612
    assert math.isclose(loss_module.forward(), predicted_loss, rel_tol=0.005)


def test_waveq_per_layer_func_type():
    tensor_4d = torch.rand((3, 3, 3, 3))
    print(WaveQLoss.waveq_loss_per_layer(tensor_4d))
    print(type(WaveQLoss.waveq_loss_per_layer(tensor_4d)))
    assert isinstance(WaveQLoss.waveq_loss_per_layer(tensor_4d), float)


def test_waveq_per_layer_func_value():
    tensor_4d = torch.tensor([[[[0.002], [0.0098]],
                               [[0.0025], [0.0021]]],
                              [[[0.0140], [0.098]],
                               [[0.033], [0.0166]]],
                              [[[0.0206], [0.0284]],
                               [[0.04], [0.029]]],
                              [[[0.0145], [0.0137]],
                               [[0.003], [0.0294]]]], dtype=torch.float32)
    tensor_4d_result = 0.04484
    assert math.isclose(WaveQLoss.waveq_loss_per_layer(tensor_4d),
                        tensor_4d_result, rel_tol=0.005)
