import numpy as np
import argparse
import torch
import math

from gnn import LinkPred

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
    parser.add_argument("--data_size", type=int, default=100, 
                        help="the dataset size. Make sure that it is a decent amount smaller than p.")
    parser.add_argument("--valid", type=float, default=0.1, 
                        help="Portion of data to be used for validation")
    parser.add_argument("--test", type=float, default=0.2, 
                        help="Portion of data to be used for testing")
    parser.add_argument("--features", type=int, default=3, 
                        help="Number of features used for each equivalence class (our features are x, x+p, x+2*p,... x+n_features*p)")

    # Training params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, 
                        help="learning rate")

    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)

    print("Prime Modulus: %d" % (args.p))
    print("Secret s: %d" % (args.s))

    lp = LinkPred(args)
    lp.compute_auc()
