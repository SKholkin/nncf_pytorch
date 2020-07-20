import pytest
from nncf.quantization.waveq_loss import WaveQLoss
import torch
import math
from torch.utils import tensorboard as tb
import matplotlib.pyplot as plt

tensor_4d_float = torch.tensor([[[[0.002], [0.0098]],
                                 [[0.0025], [0.0021]]],
                                [[[0.0140], [0.098]],
                                 [[0.033], [0.0166]]],
                                [[[0.0206], [0.0284]],
                                 [[0.04], [0.029]]],
                                [[[0.0145], [0.0137]],
                                 [[0.003], [0.0294]]]], dtype=torch.float32)


def test_waveq_loss_float_value_check():
    layers = list()
    layers.append(torch.nn.Conv2d(2, 2, 2, 2))
    layers[0].weight.data = tensor_4d_float
    layers.append(torch.nn.Conv2d(1, 1, 2, 2))
    layers[1].weight.data = torch.tensor([[[[0.033], [0.0137]], [[0.0206], [0.0166]]]])
    loss_module = WaveQLoss(convlayers=layers)
    predicted_loss = 0.05612
    assert math.isclose(loss_module.forward(), predicted_loss, rel_tol=0.005)


def test_waveq_per_layer_func_type():
    tensor_4d = torch.rand((3, 3, 3, 3))
    print(WaveQLoss.waveq_loss_per_layer_sum(tensor_4d))
    print(type(WaveQLoss.waveq_loss_per_layer_sum(tensor_4d)))
    assert isinstance(WaveQLoss.waveq_loss_per_layer_sum(tensor_4d), float)


def test_waveq_per_layer_func_value_check():
    tensor_4d_result = 0.04484
    assert math.isclose(WaveQLoss.waveq_loss_per_layer_sum(tensor_4d_float),
                        tensor_4d_result, rel_tol=0.005)


class My_model(torch.nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 6)
        self.set_rand_layers()

    def set_rand_layers(self):
        # try another distribution
        for module in self.children():
            module.weight.data = torch.randn(module.weight.data.shape)


def test_graph_view(model):
    waveq = WaveQLoss(model)
    loss_tensors = waveq.get_loss_stats_tensor(ratio=16, quant=1)
    log_dir = '/home/skholkin/projects/pycharm_storage/first_tasks/run'
    writer = tb.SummaryWriter(log_dir=log_dir)

    layer_count = 1
    weights = []
    for child in model.children():
        weights.append(child.weight.data)

    for weights_loss in zip(weights, loss_tensors):
        weights_tensor = weights_loss[0]
        loss_tensor = weights_loss[1]
        loss_tensor = torch.flatten(loss_tensor)
        weights_tensor = torch.flatten(weights_tensor)

        zipped = sorted(zip(loss_tensor, weights_tensor), key=lambda tup: tup[1])
        loss_tensor = [i[0] for i in zipped]
        weights_tensor = [i[1] for i in zipped]

        plot = plt.plot(weights_tensor, loss_tensor)
        plot.set_yscale("log")
        # ax.set_yscale('log')
        for loss_point, weight_point in zip(loss_tensor, weights_tensor):
            writer.add_scalar(f'layer {layer_count}', loss_point, weight_point)
        layer_count += 1
        plt.show()


def test_model_to_quantize_converter():
    # create fake model including quantization modules
    # to test new class features
    pass


if '__main__' == __name__:
    test_graph_view(My_model())
