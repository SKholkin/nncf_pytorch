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
        # create model through create_comp_model(config)
        PATH = args.checkpoint
        if PATH is not None:
            self.checkpoint = torch.load(PATH)
        else:
            raise argparse.ArgumentError('checkpoint not specified')
        self.log_dir = args.logdir
        # do i need is not None check?

    def print_dist(self):
        writer = tb.SummaryWriter(log_dir=self.log_dir)
        for name, tensor in self.checkpoint['state_dict'].items():
            if name.find('weight') > 0 and list(tensor.flatten().shape)[0] > 10000:
                writer.add_histogram(name, tensor, bins=500, max_bins=1000)
        if self.checkpoint['CR_loss'] is not None:
            writer.add_scalar('Compression Loss', self.checkpoint['CR_loss'], self.checkpoint['epoch'])

if __name__ == '__main__':
    tool = WeightDistributionTool(sys.argv[1:])
    tool.print_dist()
