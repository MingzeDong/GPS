from torch_geometric.utils import dropout_adj, negative_sampling
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels, normalize=True)
        self.relu = torch.nn.ReLU()
        self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels, normalize=True)

    def forward(self, node_feature, edge_index):

        tmp = self.conv1(node_feature, edge_index)
        tmp = self.relu(tmp)
        output = self.conv2(tmp, edge_index)

        return output


class GCN_1layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels=in_channels, out_channels=out_channels, normalize=True)
        self.relu = torch.nn.ReLU()

    def forward(self, node_feature, edge_index):

        tmp = self.conv1(node_feature, edge_index)
        tmp = self.relu(tmp)

        return tmp


class GCN_ensemble(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2, out_channels):
        super().__init__()
        self.conv0 = GCNConv(in_channels=in_channels, out_channels=hidden_channels1, normalize=True)
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels2, normalize=True)
        self.relu = torch.nn.ReLU()
        self.conv2 = GCNConv(in_channels=hidden_channels1, out_channels=out_channels, normalize=True)
        self.conv3 = GCNConv(in_channels=hidden_channels2, out_channels=out_channels, normalize=True)
        #self.linear = nn.Linear(in_features=2*hidden_channels2, out_features=out_channels)

    def forward(self, node_feature, edge_index, edge_index_new):

        tmp1 = self.conv0(node_feature, edge_index)
        tmp1 = self.relu(tmp1)
        tmp2 = self.conv1(node_feature, edge_index_new)
        tmp2 = self.relu(tmp2)

        output1 = self.conv2(tmp1, edge_index)
        output2 = self.conv3(tmp2, edge_index_new)
        #output = self.linear(torch.cat((output1,output2),1))
        output = output1 + output2

        return output



def train(model, data, edge_index, optimizer, loss_fn):

    loss = 0
    model.train()
    optimizer.zero_grad()
    y_pred = model(data.x,edge_index)
    if len(data.train_mask.shape)>1:
        loss = loss_fn(y_pred[data.train_mask[:,0]], data.y[data.train_mask[:,0]])
    else:
        loss = loss_fn(y_pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss

def train_ensemble(model, data, edge_index,edge_index_new, optimizer, loss_fn):

    loss = 0
    model.train()
    optimizer.zero_grad()
    y_pred = model(data.x,edge_index,edge_index_new)
    if len(data.train_mask.shape)>1:
        loss = loss_fn(y_pred[data.train_mask[:,0]], data.y[data.train_mask[:,0]])
    else:
        loss = loss_fn(y_pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(model, data, edge_index):

    accuracy_list = [0, 0, 0]
    model.eval()
    y_pred = model(data.x,edge_index)
    if len(data.train_mask.shape)>1:
        accuracy_list[0] = torch.sum(torch.argmax(y_pred,axis=1)[data.train_mask[:,0]] - data.y[data.train_mask[:,0]] == 0)/torch.sum(data.train_mask[:,0])
        accuracy_list[1] = torch.sum(torch.argmax(y_pred,axis=1)[data.val_mask[:,0]] - data.y[data.val_mask[:,0]] == 0)/torch.sum(data.val_mask[:,0])
        accuracy_list[2] = torch.sum(torch.argmax(y_pred,axis=1)[data.test_mask[:,0]] - data.y[data.test_mask[:,0]]== 0)/torch.sum(data.test_mask[:,0])
    else:
        accuracy_list[0] = torch.sum(torch.argmax(y_pred,axis=1)[data.train_mask] - data.y[data.train_mask] == 0)/torch.sum(data.train_mask)
        accuracy_list[1] = torch.sum(torch.argmax(y_pred,axis=1)[data.val_mask] - data.y[data.val_mask] == 0)/torch.sum(data.val_mask)
        accuracy_list[2] = torch.sum(torch.argmax(y_pred,axis=1)[data.test_mask] - data.y[data.test_mask]== 0)/torch.sum(data.test_mask)        
    return accuracy_list

@torch.no_grad()
def test_ensemble(model, data, edge_index, edge_index_new):

    accuracy_list = [0, 0, 0]
    model.eval()
    y_pred = model(data.x,edge_index, edge_index_new)
    if len(data.train_mask.shape)>1:
        accuracy_list[0] = torch.sum(torch.argmax(y_pred,axis=1)[data.train_mask[:,0]] - data.y[data.train_mask[:,0]] == 0)/torch.sum(data.train_mask[:,0])
        accuracy_list[1] = torch.sum(torch.argmax(y_pred,axis=1)[data.val_mask[:,0]] - data.y[data.val_mask[:,0]] == 0)/torch.sum(data.val_mask[:,0])
        accuracy_list[2] = torch.sum(torch.argmax(y_pred,axis=1)[data.test_mask[:,0]] - data.y[data.test_mask[:,0]]== 0)/torch.sum(data.test_mask[:,0])
    else:
        accuracy_list[0] = torch.sum(torch.argmax(y_pred,axis=1)[data.train_mask] - data.y[data.train_mask] == 0)/torch.sum(data.train_mask)
        accuracy_list[1] = torch.sum(torch.argmax(y_pred,axis=1)[data.val_mask] - data.y[data.val_mask] == 0)/torch.sum(data.val_mask)
        accuracy_list[2] = torch.sum(torch.argmax(y_pred,axis=1)[data.test_mask] - data.y[data.test_mask]== 0)/torch.sum(data.test_mask)        
    return accuracy_list

def GCNtrain(data,edge_index,edge_index_og=None,model=None,lr=3e-4, weight_decay=1e-4,epochs=200,seed=42,ensemble=False,plot=False,device='cpu'):
    torch.manual_seed(seed)
    if model is None:
        if ensemble:
            model = GCN_ensemble(data.x.shape[1], 100, 50, int(torch.max(data.y)+1)).to(device)
        else:
            model = GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        if ensemble:
            edge_index_og = edge_index_og.to(device)
            loss.append(train_ensemble(model, data, edge_index_og, edge_index, optimizer, loss_fn).item())
            t_acc, v_acc, te_acc = test_ensemble(model, data, edge_index_og, edge_index)
        else:
            loss.append(train(model, data, edge_index, optimizer, loss_fn).item())
            t_acc, v_acc, te_acc = test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    print("after {} epochs' training, the final test accuracy is {}".format(epochs, final_test_acc))
    if plot:
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.plot(test_acc)
        plt.show()
    return final_test_acc

