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

tensor_4d_float = torch.tensor([[[[0.002], [0.0098]],
                                 [[0.0025], [0.0021]]],
                                [[[0.0140], [0.098]],
                                 [[0.033], [0.0166]]],
                                [[[0.0206], [0.0284]],
                                 [[0.04], [0.029]]],
                                [[[0.0145], [0.0137]],
                                 [[0.003], [0.0294]]]], dtype=torch.float32)
log_dir = '/home/skholkin/projects/pycharm_storage/first_tasks/run'


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


def test_model_to_quantize_converter():
    model = BasicConvTestModel(in_channels=1, out_channels=1, kernel_size=10, weight_init=0)
    for child in model.children():
        child.weight.data.normal_()
    config = get_basic_asym_quantization_config(model_size=10)
    # how to compress weights NOT INPUT
    # cuz dummy_forward compresses only input
    # create_compressed_model_and_algo_for_test should return ctrl, model? not model , ctrl
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, QuantizationController)
    # how to implement WaveQ loss in model
    # how to create PyTorch custom loss using WaveQ
    # how to conect loss with model and exactly compressed one
    nncf_model_weights_hist(compressed_model, 'before')
    print(compression_ctrl.get_bit_stats().num_bits)
    #loss_module = WaveQLoss(list(compression_ctrl.all_quantizations.values()))
    # how to create loss in ctrl
    compressed_model.do_dummy_forward()
    loss = compression_ctrl.loss()
    print(loss)
    nncf_model_weights_hist(compressed_model, 'after')
    #loss = loss_module.forward()
    draw_waveq_loss(loss)
    assert isinstance(loss, float)


def nncf_model_weights_hist(compressed_model, name: str):
    writer = tb.SummaryWriter(log_dir=log_dir)
    count = 1
    nncf_modules = compressed_model.get_nncf_modules()
    for nncf_module in nncf_modules.values():
        writer.add_histogram(f'{name}: module {count}', nncf_module.weight.data)
        plt.hist(nncf_module.weight.data.flatten())
        count += 1
    plt.show()


def draw_waveq_loss(loss):
    writer = tb.SummaryWriter(log_dir=log_dir)
    writer.add_scalar('loss', loss)
