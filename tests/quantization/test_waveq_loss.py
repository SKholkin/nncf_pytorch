
import pytest
from nncf.quantization.waveq_loss import WaveQLoss
from nncf.quantization.algo import QuantizationController
import torch
import math
from torch.utils import tensorboard as tb
import matplotlib.pyplot as plt
from tests.test_helpers import BasicConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_algo_quantization import get_basic_quantization_config, \
    get_basic_asym_quantization_config, OnesDatasetMock
from nncf.initialization import register_default_init_args

kernel_size = 100
in_channels = 1
basic_model = BasicConvTestModel(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, weight_init=0)
criterion = torch.nn.CrossEntropyLoss()

for child in basic_model.children():
    child.weight.data.normal_()

config_4bits = get_basic_quantization_config(model_size=kernel_size)
config_4bits['compression'].update({
    "weights": {
        "mode": "symmetric",
        "signed": False,
        "per_channel": True,
        "bits": 4
    },
    "activations": {
        "mode": "symmetric",
        "signed": False,
        "bits": 4,
        "signed": True
    }
})
data_loader = torch.utils.data.DataLoader(OnesDatasetMock((in_channels ,kernel_size, kernel_size)),
                                              batch_size=1,
                                              num_workers=1,
                                              shuffle=False)

config_4bits = register_default_init_args(config_4bits, criterion, data_loader)
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


def draw_post_quant_dist(hooks_list: list):
    writer = tb.SummaryWriter(log_dir=f'{log_dir}/quant_dist')
    count = 1
    for hook in hooks_list:
        #plt.hist(hook.input_tensor.flatten(), bins=500, log=True)
        writer.add_histogram(f'quant dist: {count}', hook.input_tensor)
        count += 1
    #plt.show()
    for hook in hooks_list:
        plt.hist(hook.out_tensor.flatten(), bins=500, log=True)
    plt.show()


def draw_waveq_per_hook(hooks_list: list):
    writer = tb.SummaryWriter(log_dir=f'{log_dir}/quant_dist')
    count = 1
    for hook in hooks_list:
        plt.scatter(hook.out_tensor, WaveQLoss.waveq_loss_for_tensor(hook.out_tensor,
                                                                     quant=hook.quant_module.num_bits, scale=hook.quant_module.scale))
    plt.show()


def test_model_to_quantize_converter():
    basic_compressed_model_4bits.do_dummy_forward()
    loss = basic_compression_ctrl_4bits.loss()

    draw_post_quant_dist(basic_compression_ctrl_4bits.loss.hooks)
    draw_waveq_per_hook(basic_compression_ctrl_4bits.loss.hooks)
    assert isinstance(loss, float)


def test_waveq_quantization_period():
    basic_compressed_model_4bits.do_dummy_forward()
    hooks_list = basic_compression_ctrl_4bits.loss.hooks
    loss = 0
    for hook in hooks_list:
        loss += torch.sum(WaveQLoss.waveq_loss_for_tensor(hook.out_tensor,
                                              quant=hook.quant_module.num_bits, scale=hook.quant_module.scale))
    # zero and most likely period is ok but lower accuracy then before
    assert math.isclose(loss, 0, abs_tol=1e-8)
