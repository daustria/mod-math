import numpy as np
import argparse
import torch
import data
import math

from torch.utils.data import DataLoader
from torch.autograd import Variable

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="AI for Math")

    # Dataset params
    parser.add_argument("--problem", type=str, default="modularmult", 
                        help="the problem being solved")
    parser.add_argument("--p", type=int, default=251, 
                        help="the prime modulus")
    parser.add_argument("--s", type=int, default=3, 
                        help="the fixed secret integer in modularmult and diffiehellmanfixed")
    parser.add_argument("--g", type=int, default=113, 
                        help="the public primitive root in dlp, diffiehellman, diffiehellmanfixed")
    parser.add_argument("--base", type=int, default=8, 
                        help="the base used when we tokenize numbers")
    parser.add_argument("--test_size", type=int, default=80, 
                        help="size of the test size")
    parser.add_argument("--start_token", type=int, default=8, 
                        help="start token digit")

    # Training params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beam_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_epochs", type=int, default=2, 
                        help="number of epochs")
    parser.add_argument("--num_layers", type=int, default=2, 
                        help="number of layers in encoder/decoder")
    parser.add_argument("--lr", type=float, default=0.00005, 
                        help="learning rate")
    parser.add_argument('--do_weight_loss', action='store_true', help='A boolean flag for re-weighting loss')
    parser.add_argument('--random_emb', action='store_true', help='A boolean flag for using randomly initialized positional encodings')

    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    args.int_len = np.ceil(math.log(args.p, args.base)).astype(int)

    full_train_data, test_data = data.create_datasets(args)

    train_size = int(0.8*len(full_train_data))
    valid_size = len(full_train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(full_train_data, [train_size, valid_size])

    print("Prime Modulus: %d" % (args.p))
    print("Secret s: %d" % (args.s))
    print("Length Training Data: %d" % (len(train_data)))
    print("Length Valid Data: %d" % (len(valid_data)))
    print("Length Test Data: %d" % (len(test_data)))

    data = data.make_graph_dataset(args)
