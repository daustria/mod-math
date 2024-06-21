import torch
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import numpy as np
import data

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class LinkPred():
    def __init__(self, params):
        self.params = params
        self.device = torch.device('cpu')
        self.model = Net(1, 128, 64).to(self.device)

        # TODO : Use the user input parameters in params to set some of the parameters here, with regard to
        # things like learning rate or how we make the train/validation/test data split
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=params.lr) 

        self.criterion = torch.nn.BCEWithLogitsLoss()

        train_data, val_data, test_data = data.make_graph_dataset(params)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def negative_sampling(self, dataset):
        n_edges_add = dataset.edge_index[0].size(0)

        # Use seed+1 since we already used seed to generate the graph datasets
        neg_edge_index = data.generate_negative_edges(n_edges_add, self.params, self.params.seed+1)

        new_edge_label_index = torch.cat([dataset.edge_label_index, neg_edge_index], dim=-1)

        new_edge_label = torch.cat([dataset.edge_label, dataset.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)

        return new_edge_label_index, new_edge_label

    def train(self):
        self.model.train()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)

        # These are the same as their versions in train_data but with negative edges added to them.
        new_edge_label_index, new_edge_label = self.negative_sampling(self.train_data)

        # Now we get our model to try and classify these edges, which have
        # our old edges as well as the new ones we negative sampled.
        out = self.model.decode(z, new_edge_label_index).view(-1)
        loss = self.criterion(out, new_edge_label.float())
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        out = self.model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    def compute_auc(self):
        best_val_auc = final_test_auc = 0
        for epoch in range(1, self.params.num_epochs+1):
            loss = self.train()
            val_auc = self.test(self.val_data)
            test_auc = self.test(self.test_data)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
                    f'Test: {test_auc:.4f}')

        print(f'Final Test: {final_test_auc:.4f}')

        z = self.model.encode(self.test_data.x, self.test_data.edge_index)
        final_edge_index = self.model.decode_all(z)
