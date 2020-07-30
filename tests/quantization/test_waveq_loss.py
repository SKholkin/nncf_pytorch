import pytest
from nncf.quantization.waveq_loss import WaveQLoss, get_quant_module_params
from nncf.quantization.algo import QuantizationController
import torch
import math
from torch.utils import tensorboard as tb
import matplotlib.pyplot as plt
from tests.test_helpers import BasicConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_algo_quantization import get_basic_quantization_config, \
    get_basic_asym_quantization_config, OnesDatasetMock
from nncf.initialization import register_default_init_args

kernel_size = 10
in_channels = 1
basic_model = BasicConvTestModel(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, weight_init=0)
criterion = torch.nn.CrossEntropyLoss()

for child in basic_model.children():
    child.weight.data.normal_()

config_4bits = get_basic_quantization_config(model_size=kernel_size)
config_4bits['compression'].update({
    "waveq": True,
    "weights": {
        "mode": "symmetric",
        "signed": True,
        "per_channel": True,
        "bits": 3
    },
    "activations": {
        "mode": "symmetric",
        "signed": True,
        "bits": 3
    }
})
# config init changes
data_loader = torch.utils.data.DataLoader(OnesDatasetMock((in_channels, kernel_size, kernel_size)),
                                          batch_size=1,
                                          num_workers=1,
                                          shuffle=False)

config_4bits = register_default_init_args(config_4bits, criterion, data_loader)
basic_compressed_model_4bits, basic_compression_ctrl_4bits = \
    create_compressed_model_and_algo_for_test(basic_model, config_4bits)
log_dir = '/home/skholkin/projects/pycharm_storage/tb/bucket'


def draw_post_quant_dist(hooks_list: list):
    writer = tb.SummaryWriter(log_dir=f'{log_dir}/quant_dist')
    count = 1
    for hook in hooks_list:
        # plt.hist(hook.input_tensor.flatten(), bins=500, log=True)
        count += 1
    # plt.show()
    #for hook in hooks_list:
        #plt.hist(hook.out_tensor.flatten(), bins=500, log=True)


def draw_waveq_graphic(hooks):
    input_low, input_range, levels = get_quant_module_params(hooks[0].quant_module)
    waveq_graphic(levels=levels, input_low=input_low, input_range=input_range)


def waveq_graphic(levels=16, input_low=0, input_range=1):
    # tensor = torch.arange(level_low, level_high, step=1e-3)
    for i in range(0, levels):
        plt.axvline(input_low + input_range * i / (levels - 1))
    tensor = torch.arange(start=float(input_low), end=float(input_range + input_low), step=1e-3)
    waveq_loss = WaveQLoss.waveq_loss_for_tensor(tensor, levels=levels, input_low=input_low, input_range=input_range)
    plt.plot(tensor.data.flatten(), waveq_loss.data.flatten())
    plt.show()


def test_model_to_quantize_converter():
    basic_compressed_model_4bits.do_dummy_forward()
    loss = basic_compression_ctrl_4bits.loss()

    draw_post_quant_dist(basic_compression_ctrl_4bits.loss.hooks)
    draw_waveq_graphic(hooks=basic_compression_ctrl_4bits.loss.hooks)
    # draw_waveq_per_hook(basic_compression_ctrl_4bits.loss.hooks)
    print(loss)
    loss.backward()


def test_waveq_quantization_period():
    basic_compressed_model_4bits.do_dummy_forward()
    hooks_list = basic_compression_ctrl_4bits.loss.hooks
    loss = 0

    for hook in hooks_list:
        input_low, input_range, levels = get_quant_module_params(hook.quant_module)
        loss += torch.sum(WaveQLoss.waveq_loss_for_tensor(hook.out_tensor,
                                                          levels=levels,
                                                          input_low=input_low,
                                                          input_range=input_range))
    print(loss)
    assert math.isclose(float(loss.data), 0, abs_tol=5e-11)
