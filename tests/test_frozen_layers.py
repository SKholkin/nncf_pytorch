import pytest
import torch
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.pruning.helpers import get_basic_pruning_config, BigPruningTestModel
from tests.sparsity.rb.test_algo import get_basic_sparsity_config


def get_pruning_config(input_sample_size=None, prune_last_conv=False):
    config = get_basic_pruning_config(input_sample_size)
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['prune_last_conv'] = prune_last_conv
    return config


@pytest.mark.parametrize('config,model', [(get_quantization_config_without_range_init(), TwoConvTestModel()),
                                          (get_pruning_config(prune_last_conv=True), TwoConvTestModel()),
                                          (get_pruning_config([1, 1, 10, 10]), BigPruningTestModel()),
                                          (get_basic_sparsity_config(), TwoConvTestModel())])
def test_frozen_layers(config, model):
    # freeze first conv
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.weight.requires_grad = False
            break

    compressed_model, compression_algo = create_compressed_model_and_algo_for_test(model, config)
    for scope, module in compressed_model.get_nncf_modules().items():
        if not module.weight.requires_grad:
            assert len(module.pre_ops) == 0
