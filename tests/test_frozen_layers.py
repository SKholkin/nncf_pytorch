import pytest
import torch

from examples.common.execution import prepare_model_for_execution, get_device
from examples.common.model_loader import load_resuming_model_state_dict_and_checkpoint_from_path
from examples.common.sample_config import SampleConfig
from examples.object_detection.main import create_dataloaders
from examples.object_detection.model import build_ssd
from nncf import load_state, create_compressed_model, NNCFConfig, register_default_init_args
from nncf.dynamic_graph.context import ScopeElement, Scope
from nncf.dynamic_graph.graph_builder import create_input_infos
from tests.conftest import EXAMPLES_DIR, PROJECT_ROOT, TEST_ROOT
from nncf.dynamic_graph.transform_graph import replace_modules_by_nncf_modules
from nncf.utils import get_all_modules_by_type, get_all_modules
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init


def create_model(config, resuming_model_sd: dict = None):
    input_info_list = create_input_infos(config.nncf_config)
    image_size = input_info_list[0].shape[-1]
    ssd_net = build_ssd(config.model, config.ssd_params, image_size, config.num_classes, config)
    weights = config.get('weights')
    if weights:
        sd = torch.load(weights, map_location='cpu')
        load_state(ssd_net, sd)
    # set requires_grad=False to some layers (weights)

    ssd_net.to(config.device)
    return ssd_net


def get_model():
    pass


def create_config():
    path = PROJECT_ROOT.joinpath("examples", "object_detection", "configs", "ssd300_vgg_voc_int8.json")
    nncf_config = NNCFConfig.from_json(path)
    config = SampleConfig.from_json(path)
    config.dataset_dir = '/home/skholkin/datasets/VOCdevkit/'
    config.resuming_checkpoint_path = '/home/skholkin/projects/results/main_checkpoints/ssd300_int8_voc12.pth'
    config.nncf_config = nncf_config
    return config


# ToDo: maybe use mock datasets
# ToDo: try pruning exception
def frozen_layers_ssd():
    current_gpu = 0
    config = create_config()
    config.current_gpu = current_gpu
    config.device = get_device(config)
    resuming_checkpoint_path = config.resuming_checkpoint_path
    # test_data_loader, train_data_loader, init_data_loader = create_dataloaders(config)
    # nncf_config = register_default_init_args(config.nncf_config, init_data_loader, criterion, criterion_fn, config.device)
    resuming_model_sd = None
    resuming_model_sd, resuming_checkpoint = load_resuming_model_state_dict_and_checkpoint_from_path(
        resuming_checkpoint_path)
    model = create_model(config, resuming_model_sd)
    model = TwoConvTestModel()

    children = list(model.children())
    basenet_children = list(children[0].children())
    child_params = list(basenet_children[0].parameters())

    name_params = list(model.named_parameters())

    for param in basenet_children[0].parameters():
        param.requires_grad = False

    basenet_children[0] = replace_modules_by_nncf_modules(model)

    all_modules_by_type = get_all_modules_by_type(model)

    # for name, module in model.named_children():
    # search for requires grad = false modules
    # scope = ScopeElement(module.__class__.__name__, name)
    # ignored_scoped.append(str(scope))

    compression_ctrl, compressed_model = create_compressed_model(model, config.nncf_config, resuming_model_sd)
    compressed_model, _ = prepare_model_for_execution(compressed_model, config)
    comp_model_name_params = list(compressed_model.named_parameters())
    compressed_model.train()


def test_frozen_layers():
    model = TwoConvTestModel()
    config = get_quantization_config_without_range_init()

    # first conv frozen
    for param in model.features[0][0].parameters():
        param.requires_grad = False

    compressed_model, compression_algo = create_compressed_model_and_algo_for_test(model, config)
    for scope, module in compressed_model.get_nncf_modules().items():
        for param in module.parameters():
            if param.requires_grad == False:
                assert len(module.pre_ops) == 0
