from typing import Callable

import pytest
from torch import nn

from nncf import NNCFConfig
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.pruning.helpers import get_basic_pruning_config, BigPruningTestModel
from tests.sparsity.rb.test_algo import get_basic_sparsity_config
from tests.test_nncf_network import TwoConvTestModelWithUserModule, ModuleOfUser
from typing import NamedTuple


def get_pruning_config(input_sample_size=None, prune_first_conv=False):
    config = get_basic_pruning_config(input_sample_size)
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['prune_first_conv'] = prune_first_conv
    return config


def freeze_module(model, module_to_freeze, freeze_count):
    counter = 1
    for module in model.modules():
        if isinstance(module, module_to_freeze):
            if counter < freeze_count:
                counter += 1
                continue
            module.weight.requires_grad = False
            break


class FrozenLayersTestStruct(NamedTuple):
    config: NNCFConfig = get_quantization_config_without_range_init()
    model_creator: Callable[[], nn.Module] = TwoConvTestModel
    module_to_freeze: nn.Module = nn.Conv2d
    freeze_count: int = 1

    def __str__(self):
        return '_'.join(['frozen', str(self.freeze_count), self.module_to_freeze.__name__,
                         self.config['compression']['algorithm'], self.model_creator.__name__])


TEST_PARAMS = [
    FrozenLayersTestStruct(),
    FrozenLayersTestStruct(model_creator=TwoConvTestModelWithUserModule, module_to_freeze=ModuleOfUser),
    FrozenLayersTestStruct(config=get_basic_sparsity_config(), freeze_count=2),
    FrozenLayersTestStruct(config=get_pruning_config(),
                           model_creator=TwoConvTestModel, freeze_count=1),
    FrozenLayersTestStruct(config=get_pruning_config([1, 1, 10, 10]),
                           model_creator=BigPruningTestModel, freeze_count=2),
    FrozenLayersTestStruct(config=get_pruning_config(),
                           model_creator=TwoConvTestModelWithUserModule, freeze_count=2)
]


@pytest.mark.parametrize('params', TEST_PARAMS, ids=[str(p) for p in TEST_PARAMS])
def test_frozen_layers(params):
    model = params.model_creator()
    config = params.config

    freeze_module(model, params.module_to_freeze, params.freeze_count)

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    for module in compressed_model.get_nncf_modules().values():
        if not module.weight.requires_grad:
            assert len(module.pre_ops) == 0
