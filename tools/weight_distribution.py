import argparse
import sys
import torch
from torch.utils import tensorboard as tb


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
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


class WeightDistributionTool:

    def __init__(self, argv):
        parser = get_parser()
        args = parser.parse_args(args=argv)
        # checkpoint_path -> state_dict
        # state_dict -> weights
        # weights -> tensorboard hist
        PATH = args.checkpoint
        if PATH is not None:
            self.checkpoint = torch.load(PATH)
        else:
            raise argparse.ArgumentError('checkpoint not specified')
        self.log_dir = args.logdir
        # do i need is not None check?
        pass

    def print_dist(self):
        writer = tb.SummaryWriter(log_dir=self.log_dir)
        for name, tensor in self.checkpoint['state_dict'].items():
            if name.find('weight') > 0:
                writer.add_histogram(name, tensor)


if __name__ == '__main__':
    tool = WeightDistributionTool(sys.argv[1:])
    tool.print_dist()
