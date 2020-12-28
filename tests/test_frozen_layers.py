import pytest
import torch
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.pruning.helpers import get_basic_pruning_config, BigPruningTestModel, TestModelDiffConvs


def get_pruning_config():
    config = get_basic_pruning_config(input_sample_size=[1, 1, 6, 6])
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['num_init_steps'] = 0
    config['compression']['params']['schedule'] = "baseline"
    config['compression']['params']['pruning_target'] = 0.3
    return config


def test_frozen_layers():
    model = TwoConvTestModel()
    config = get_quantization_config_without_range_init()

    # first conv frozen
    for module in model.modules():
        if module.__class__ == torch.nn.Conv2d:
            for param in module.parameters():
                param.requires_grad = False
            break

    compressed_model, compression_algo = create_compressed_model_and_algo_for_test(model, config)
    for scope, module in compressed_model.get_nncf_modules().items():
        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                assert len(module.pre_ops) == 0
