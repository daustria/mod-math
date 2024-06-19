import torch
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

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
    def __init__(self, params, dataset):
        self.params = params
        self.device = torch.device('cpu')
        self.model = Net(params.p, 128, 64).to(self.device)

        # TODO : Use the user input parameters in params to set some of the parameters here, with regard to
        # things like learning rate or how we make the train/validation/test data split

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        transform = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, add_negative_train_samples=False)

        train_data, val_data, test_data = transform(dataset)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def negative_sampling(self, dataset):
        # TODO
        pass

    def train(self):
        self.model.train()
        z = self.model.encode(train_data.x, train_data.edge_index)

        neg_edge_index = self.negative_sampling() # TODO

        # We add these new edges to edge_label_index 
        edge_label_index = torch.cat(
                [self.train_data.edge_label_index, neg_edge_index],
                dim=-1,
                )

        # Label them 0. the edges that are 'actually' in the graph have label 1
        edge_label = torch.cat([
            self.train_data.edge_label,
            self.train_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)

        # Now we get our model to try and classify these edges, which have
        # our old edges as well as the new ones we negative sampled.
        out = self.model.decode(z, edge_label_index).view(-1)
        loss = self.criterion(out, edge_label)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        z = model.encode(data.x, data.edge_index)
        out = self.model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    def compute_auc(self):
        best_val_auc = final_test_auc = 0
        for epoch in range(1, 101):
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
