import argparse
import sys
import torch
from torch.utils import tensorboard as tb
import re
import matplotlib.pyplot as plt
import os
from nncf.nncf_network import NNCFNetwork


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        metavar='mode',
        type=str,
        default='checkpoint',
        help='Specifies mode: checkpoint/log')
    parser.add_argument(
        "--path",
        metavar='PATH',
        type=str,
        default=None,
        help="Specifies the directory for the trained model checkpoints to be loaded from")
    parser.add_argument(
        "--logdir",
        metavar='PATH',
        type=str,
        default='/home/skholkin/projects/pycharm_storage/tb/bucket/plots',
        help="Specifies the directory graphics storage")

    return parser


LOG = 'log'
CHECKPOINT = 'checkpoint'

log_dir_def = '/home/skholkin/projects/pycharm_storage/tb/bucket/plots'

meaningful_layers_list_resnet_cifar = ['ResNet/NNCFLinear[fc]',
                                       'ResNet/Sequential[layer1]/Bottleneck[2]/NNCFConv2d[conv2]',
                                       'ResNet/Sequential[layer2]/Bottleneck[3]/NNCFConv2d[conv2]',
                                       'ResNet/Sequential[layer4]/Bottleneck[0]/NNCFConv2d[conv3]',
                                       'ResNet/Sequential[layer2]/Bottleneck[0]/NNCFConv2d[conv2]']

meaningful_layers_list_mobilenetv2_cifar = ['MobileNetV2/Sequential[classifier]/NNCFLinear[1]',
                                            'MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/NNCFConv2d[2]',
                                            'MobileNetV2/Sequential[features]/InvertedResidual[7]/Sequential[conv]/NNCFConv2d[2]',
                                            'MobileNetV2/Sequential[features]/InvertedResidual[12]/Sequential[conv]/NNCFConv2d[2]',
                                            'MobileNetV2/Sequential[features]/InvertedResidual[15]/Sequential[conv]/NNCFConv2d[2]',
                                            'MobileNetV2/Sequential[features]/InvertedResidual[12]/Sequential[conv]/ConvBNReLU[1]/NNCFConv2d[0]',
                                            'MobileNetV2/Sequential[features]/InvertedResidual[14]/Sequential[conv]/ConvBNReLU[1]/NNCFConv2d[0]']


def print_weight_dist(model: NNCFNetwork, log_dir=log_dir_def, name=None):
    log_dir = deal_with_paths(log_dir, name)
    pairs = get_module_quantizer_pairs(model)
    for nncf_module_dict, quantizer in pairs:
        if nncf_module_dict['scope'] in meaningful_layers_list_resnet_cifar:
            input_low, input_range = quantizer.calculate_inputs()
            scope = nncf_module_dict['scope'].replace('/', '.')
            path = os.path.join(log_dir, scope)
            print_pair(path, nncf_module_dict['nncf_module'].weight, input_low, input_range, quantizer.levels)


def deal_with_paths(log_dir, name):
    log_dir = os.path.join(log_dir, 'plots')
    try:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
    except:
        BaseException('data race')
    try:
        if name is not None:
            log_dir = os.path.join(log_dir, name)
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
    except:
        BaseException('data race')
    return log_dir


def get_module_quantizer_pairs(model: NNCFNetwork):  # list[]
    pairs = []
    if not isinstance(model, NNCFNetwork):
        model = model.module
    if not isinstance(model, NNCFNetwork):
        return []
    for scope, nncf_module in model.get_nncf_modules().items():
        nncf_module_dict = {'scope': str(scope), 'nncf_module': nncf_module}
        for quantizer in nncf_module.pre_ops.values():
            pairs.append((nncf_module_dict, quantizer.operand))
    return pairs


def print_pair(path, weights, input_low, input_range, levels):
    # TODO: deal with levels - 1 problem and another quant problems
    with torch.no_grad():
        a = weights.cpu().numpy().flatten()
        plt.hist(a, bins=200,
                 range=(float(input_low * 1.5), float(input_low + input_range) * 1.5))
        plt.axvline(input_low, color='r')
        plt.axvline(input_low + input_range, color='r')
        for mul in range(levels):
            pass
            plt.axvline(input_low + input_range * (mul / (levels - 1)), linewidth=3, color='black')
        try:
            save_plt(path)
        except:
            BaseException('data race')
        plt.clf()


def save_plt(name: str, log_dir=log_dir_def):
    path = os.path.join(log_dir, name)
    plt.savefig(f'{path}.pdf', format='pdf')


class WeightDistributionTool:

    def __init__(self, argv):
        parser = get_parser()
        args = parser.parse_args(args=argv)
        # create model through create_comp_model(config)
        PATH = args.path
        self.mode = args.mode
        self.log_dir = args.logdir
        self.writer = tb.SummaryWriter(log_dir=self.log_dir)
        self.checkpoint = None
        self.log_input = None
        if PATH is not None:
            try:
                if self.mode == 'checkpoint':
                    self.checkpoint = torch.load(PATH)
                else:
                    with open(PATH) as log_input:
                        self.log_input = log_input.read()
            except:
                raise FileNotFoundError('wrong PATH')
        else:
            raise argparse.ArgumentError('checkpoint not specified')
        # do i need is not None check?

    def show(self):
        if self.mode == LOG:
            self.print_acc_dynamics()
            self.print_loss_dynamics()
        elif self.mode == CHECKPOINT:
            self.print_dist_checkpoint()
        self.writer.close()

    def print_acc_dynamics(self):
        acc_list = self.find_acc_in_log()
        epoch_count = 0
        for pair in acc_list:
            self.writer.add_scalar('Acc@1', pair[0], epoch_count)
            self.writer.add_scalar('Acc@5', pair[1], epoch_count)
            epoch_count += 1

    def find_acc_in_log(self):  # tuple(Acc@1, Acc@5)
        acc_list = list()
        matches_acc1 = re.findall(r'Acc@1 (\d*\.\d*)', self.log_input)
        matches_acc5 = re.findall(r'Acc@5 (\d*\.\d*)', self.log_input)
        matches_acc1 = [float(match) for match in matches_acc1]
        matches_acc5 = [float(match) for match in matches_acc5]
        return list(zip(matches_acc1, matches_acc5))

    def print_loss_dynamics(self):
        self.print_CE_loss_dynamics()
        self.print_CR_loss_dynamics()

    def print_CE_loss_dynamics(self):
        pass

    def print_CR_loss_dynamics(self):
        pass

    def print_dist_checkpoint(self):
        for name, tensor in self.checkpoint['state_dict'].items():
            name = name.replace('nncf_module.', '')
            if name.find('weight') > 0 and list(tensor.flatten().shape)[0] > 10000:
                # self.writer.add_histogram_raw()
                self.writer.add_histogram(name, tensor, bins=1500)
                # plt.hist(tensor.cpu().flatten(), bins=1500)
                # plt.show()


if __name__ == '__main__':
    tool = WeightDistributionTool(sys.argv[1:])
    tool.show()
