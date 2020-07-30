import argparse
import sys
import torch
from torch.utils import tensorboard as tb
import re


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
        default=None,
        help="Specifies the directory graphics storage")

    return parser


LOG = 'log'
CHECKPOINT = 'checkpoint'


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
            self.print_dist()
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

    def print_dist(self):
        for name, tensor in self.checkpoint['state_dict'].items():
            name = name.replace('nncf_module.', '')
            if name.find('weight') > 0 and list(tensor.flatten().shape)[0] > 10000:
                self.writer.add_histogram(name, tensor, bins=500, max_bins=1000)


if __name__ == '__main__':
    tool = WeightDistributionTool(sys.argv[1:])
    tool.show()
