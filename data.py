import numpy as np
import torch
from torch_geometric.data import Data

class ModularMult(object):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.rng = np.random.RandomState(params.seed)
        self.p = params.p

    def get_sample(self, x):
        y = (x * self.params.s) % self.p
        # x_ = write_in_base([x], self.int_len, self.base)
        # y_ = write_in_base([y], self.int_len, self.base)
        # return x_, y_
        return x, y

    def gen_train_x(self):
        return self.rng.randint(self.p)

def generate_negative_edges(n_edges, params, seed):
    rng = np.random.RandomState(seed)

    # This could potentially loop of n_edges is big and p is small. Let's make sure that doesn't happen
    p = params.p
    s = params.s
    assert(n_edges < p*p)

    processed = { i:set() for i in range(p) }

    k = 0
    negative_edges = []

    while k < n_edges:
        a = rng.randint(p)
        b = rng.randint(p)

        if (b % p) == (a % p):
            # Disallow edges that are self loops
            continue
        elif (a*s) % p == b % p:
            # Disallow edges that are not negative (these are actually in the graph)
            continue
        elif a in processed[b] or b in processed[a]:
            # We already processed this edge
            continue
        else:
            processed[a].add(b)
            negative_edges.append([a,b])
            negative_edges.append([b,a])
            k += 1

    negative_edges = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()
    return negative_edges

def make_graph_dataset(params, max_train_size = 1000):
    gen = ModularMult(params)
    edge_index = []
    processed = [0 for _ in range(params.p)]

    for j in range(max_train_size):
        x = gen.gen_train_x()

        if processed[x]:
            continue

        x,y = gen.get_sample(x)

        processed[x] = 1
        processed[y] = 1

        edge_index.append([x,y])

        if x == 0:
            continue

        edge_index.append([y,x])

    n = len(edge_index)

    valid_size = 2*int(n*params.valid)

    test_size = 2*int(n*params.test)

    valid_edge_index = edge_index[:valid_size]
    test_edge_index = edge_index[valid_size:valid_size+test_size]
    train_edge_index = edge_index[valid_size+test_size:]

    valid_edge_index = torch.tensor(valid_edge_index, dtype=torch.long).t().contiguous()
    test_edge_index = torch.tensor(test_edge_index, dtype=torch.long).t().contiguous()
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()

    negative_edges = generate_negative_edges((valid_size+test_size) / 2, params, params.seed)

    x = torch.tensor([[i] for i in range(params.p)], dtype=torch.float)
    # Train data will have no negative edges in its graph, since we will add new negative edges to it every training epoch
    train_data = Data(x=x, 
                        edge_index=train_edge_index, 
                        edge_label=torch.ones(train_edge_index[0].size(dim=0), dtype=torch.long), 
                        edge_label_index=train_edge_index)

    # Valid and test data will have negative edges fixed.
    valid_data = construct_data_with_negative_edges(x, valid_edge_index, negative_edges[:, :valid_size])
    test_data = construct_data_with_negative_edges(x, test_edge_index, negative_edges[:, :valid_size])

    return train_data, valid_data, test_data

def construct_data_with_negative_edges(x, edge_index, negative_edges): 
    edge_label_index = torch.cat([edge_index, negative_edges], dim=-1)
    edge_label = torch.ones(edge_index[0].size(dim=0), dtype=torch.long)
    edge_label = torch.cat([edge_label, edge_label.new_zeros(negative_edges.size(1))], dim=0)
    return Data(x=x, edge_index=edge_index, edge_label=edge_label, edge_label_index=edge_label_index)

