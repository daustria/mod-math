import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class ModularMult(object):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.rng = np.random.RandomState(params.seed)
        self.test_data = self.rng.choice(params.p, params.test_size, replace=False)
        self.base, self.int_len = params.base, params.int_len
        self.p = params.p

    def get_sample(self, x):
        y = (x * self.params.s) % self.p
        # x_ = write_in_base([x], self.int_len, self.base)
        # y_ = write_in_base([y], self.int_len, self.base)
        # return x_, y_

        return x, y

    def gen_train_x(self):
        return self.rng.randint(self.p)

def make_graph_dataset(params, max_train_size = 1000):
    # What do we need to include in our Data object?

    # Dont include any negative edges yet
    # edge_label: Should be 1 all the time.
    # edge_label_index:  Should this include the edges actually in our graph? Yeah sure. Let's just make it same as edge_index for now.
    # edge_index: edges in our graph.
    gen = ModularMult(params)
    # test_set = set(gen.test_data)
    edge_index = []
    processed = [0 for _ in range(params.p)]

    for j in range(max_train_size):
        x = gen.gen_train_x()

        if processed[x]:
            continue

        processed[x] = 1

        x,y = gen.get_sample(x)

        edge_index.append([x,y])

        if x == 0:
            continue

        edge_index.append([y,x])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor([[i] for i in range(params.p)], dtype=torch.long)

    edge_label = torch.ones(edge_index[0].size(dim=0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_label=edge_label)
    print(data)
    data.validate(raise_on_error=True)
    return data

def generate_negative_sample(params):
    gen = ModularMult(params)
    a = gen.gen_train_x()
    b = gen.gen_train_x

    while (b * params.p) % p == a:
        b = gen.gen_train_x

    return torch.tensor([[a,b], [b,a]], dtype=torch.long)

