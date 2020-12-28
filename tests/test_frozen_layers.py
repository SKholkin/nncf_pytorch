import pytest
import torch
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.pruning.helpers import get_basic_pruning_config
from tests.sparsity.rb.test_algo import get_basic_sparsity_config


@pytest.mark.parametrize('config', [get_quantization_config_without_range_init(), get_basic_pruning_config(), get_basic_sparsity_config()])
def test_frozen_layers(config):
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
