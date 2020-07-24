import pytest
from nncf.quantization.waveq_loss import WaveQLoss
from nncf.quantization.algo import QuantizationController
import torch
import math
from torch.utils import tensorboard as tb
import matplotlib.pyplot as plt
from tests.test_helpers import BasicConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_algo_quantization import get_basic_quantization_config, \
    get_basic_asym_quantization_config

basic_model = BasicConvTestModel(in_channels=1, out_channels=1, kernel_size=100, weight_init=0)
for child in basic_model.children():
    child.weight.data.normal_()
config_4bits = get_basic_quantization_config(model_size=100)
config_4bits['compression'].update({
    "weights": {
        "mode": "asymmetric",
        "per_channel": True,
        "bits": 4
    },
    "activations": {
        "mode": "asymmetric",
        "bits": 4,
        "signed": True,
    }
})
basic_compressed_model_4bits, basic_compression_ctrl_4bits = \
        create_compressed_model_and_algo_for_test(basic_model, config_4bits)
tensor_4d_float = torch.tensor([[[[0.002], [0.0098]],
                                 [[0.0025], [0.0021]]],
                                [[[0.0140], [0.098]],
                                 [[0.033], [0.0166]]],
                                [[[0.0206], [0.0284]],
                                 [[0.04], [0.029]]],
                                [[[0.0145], [0.0137]],
                                 [[0.003], [0.0294]]]], dtype=torch.float32)
log_dir = '/home/skholkin/projects/pycharm_storage/tb/bucket'


def test_waveq_loss_float_value_check():
    # change model
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


def draw_post_quant_dist(hooks_list: list):
    writer = tb.SummaryWriter(log_dir=f'{log_dir}/quant_dist')
    count = 1
    for hook in hooks_list:
        plt.hist(hook.out_tensor.flatten())
        writer.add_histogram(f'quant dist: {count}', hook.out_tensor)
        count += 1
    plt.show()


def draw_waveq_per_hook(hooks_list: list):
    writer = tb.SummaryWriter(log_dir=f'{log_dir}/quant_dist')
    count = 1
    for hook in hooks_list:
        plt.scatter(hook.out_tensor, WaveQLoss.waveq_loss_for_tensor(hook.out_tensor))
    plt.show()


def test_model_to_quantize_converter():
    basic_compressed_model_4bits.do_dummy_forward()
    loss = basic_compression_ctrl_4bits.loss()

    nncf_model_weights_hist(basic_compressed_model_4bits)
    draw_post_quant_dist(basic_compression_ctrl_4bits.loss.post_hooks)
    draw_waveq_per_hook(basic_compression_ctrl_4bits.loss.post_hooks)
    print(loss)
    assert isinstance(loss, float)

def test_waveq_quantization_period():
    basic_compressed_model_4bits.do_dummy_forward()
    loss = basic_compression_ctrl_4bits.loss()
    assert math.isclose(loss, 0)

def nncf_model_weights_hist(compressed_model):
    writer = tb.SummaryWriter(log_dir=f'{log_dir}/quant_dist')
    count = 1
    nncf_modules = compressed_model.get_nncf_modules()
    # name by scopes
    for nncf_module in nncf_modules.values():
        writer.add_histogram(f': module {count}', nncf_module.weight.data, bins=300)
        count += 1

