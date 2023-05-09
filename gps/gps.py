import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GAE
from torch_geometric.utils import train_test_split_edges, to_dense_adj
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from torch_geometric.nn import GCNConv
#import scanpy as sc

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

class VGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.grad1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        self.relu = torch.nn.ReLU()
        self.grad2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index):

        m = self.grad1(x, edge_index)
        m = self.relu(m)
        m = self.grad2(m, edge_index)
        m = self.sigmoid(m)

        return m

class LinearGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.grad1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.grad2 = nn.Linear(hidden_channels, out_channels)
        #########################################

    def forward(self, x, edge_index):

        m = self.grad1(x)
        m = self.relu(m)
        m = self.grad2(m)

        return m

class NoneGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.grad1 = nn.Linear(in_channels, out_channels)
    def forward(self, x, edge_index):

        m = self.grad1(x)

        return m
    

class ATDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1l = nn.Linear(in_channels, hidden_channels)
        self.lin1r = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, z, edge_index, sigmoid=True):
        #z = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        z = self.lin1l(z[edge_index[0]]) + self.lin1r(z[edge_index[1]])
        z = self.leakyrelu(z)
        z = self.lin2(z)
        return torch.sigmoid(z) if sigmoid else z

    def forward_all(self, z, sigmoid=True):
        z_ = self.lin1l(z)[:,None,:] + self.lin1r(z)[None,:,:]
        z_ = self.leakyrelu(z_)
        z_ = self.lin2(z_)
        return torch.sigmoid(z_) if sigmoid else z_
    
    
class LogisticDecoder(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.lin = nn.Linear(out_channels, 1)

    def forward(self, z, edge_index, sigmoid=True):
        #z = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        z = z[edge_index[0]] + z[edge_index[1]]
        z = self.lin(z)
        return torch.sigmoid(z) if sigmoid else z

    def forward_all(self, z, sigmoid=True):
        z_ = z[:,None,:]+z[None,:,:]
        z_ = self.lin(z_)
        return torch.sigmoid(z_) if sigmoid else z_


def edge_train(model, data, optimizer):

    loss = 0
    model.train()
    optimizer.zero_grad()

    z1 = model.encode(data.x,data.edge_index[:,data.edge_label])
    neg_edges = negative_sampling(data.edge_index,data.x.shape[0])
    loss1 = model.recon_loss(z1,data.edge_index[:,~data.edge_label],neg_edge_index=neg_edges)
    z2 = model.encode(data.x,data.edge_index[:,~data.edge_label])
    loss2 = model.recon_loss(z2,data.edge_index[:,data.edge_label],neg_edge_index=neg_edges)
    #loss = loss + (1/data.num_nodes) * model.kl_loss()
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()

    return loss

def node_train(model, data, y1, y2, optimizer):

    loss = 0
    model.train()
    optimizer.zero_grad()
    loss_func = nn.MSELoss()

    z1 = model(data.x,data.edge_index[:,data.edge_label])
    loss1 = loss_func(z1,y1)
    z2 = model(data.x,data.edge_index[:,~data.edge_label])
    loss2 = loss_func(z2,y2)
    #loss = loss + (1/data.num_nodes) * model.kl_loss()
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()

    return loss


def edge_test(model, data):

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x,data.train_pos_edge_index)

    return model.test(z,data.test_pos_edge_index,data.test_neg_edge_index)

def gps_logistic(data,model=None,lr=5e-4,weight_decay=1e-2,l=0,epochs=500,n_neighbor='og',seed=42,device='cpu',verbose=True):
    torch.manual_seed(seed)
    idx = (torch.rand(data.edge_index.shape[1])>0.5).to(device)
    data_tmp = data.clone()

    ### For large datasets (Pubmed), use 8-head attention instead of 32
    if model is None:
        model = GAE(encoder = SGE(data.x.shape[1],10), decoder = LogisticDecoder(10)).to(device)

    data_tmp = data_tmp.to(device)
    edge_index = data_tmp.edge_index.clone()

    data_tmp.train_mask = data_tmp.val_mask = data_tmp.test_mask = None

    data_tmp.edge_label = idx

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    final_test_acc = 0
    loss = []
    test_auc = []
    test_ap = []
    for epoch in range(1, epochs + 1):
        loss.append(edge_train(model, data_tmp, optimizer).item())
    model.eval()
    z = model.encode(data.x.to(device),edge_index[:,data_tmp.edge_label]) +  model.encode(data.x.to(device),edge_index[:,~data_tmp.edge_label])
    E = torch.squeeze(model.decoder.forward_all(z))
    #E = E + E.T
    S = to_dense_adj(edge_index).squeeze()
    E = (1-l)*E + l*S
    if n_neighbor == 'og':
        trueE = torch.sum(S,dim=1) / S.shape[1]
        e0 = torch.quantile(E,1-2*torch.squeeze(trueE))
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    else:
        top_values, top_indices = torch.topk(E, k=n_neighbor, dim=0)
        edge_index_new = torch.stack([top_indices.flatten(), torch.arange(E.shape[0]).to(device).repeat(top_indices.shape[0]).flatten()], dim=1).T

    return edge_index_new, E


def gps_linkprediction(data,model=None,lr=5e-4,weight_decay=1e-2,epochs=500,n_neighbor='og',seed=42,device='cpu',verbose=True):
    torch.manual_seed(seed)
    idx = (torch.rand(data.edge_index.shape[1])>0.5).to(device)
    data_tmp = data.clone()

    ### For large datasets (Pubmed), use 8-head attention instead of 32
    if model is None:
        model = GAE(encoder = VGE(data.x.shape[1],512,128), decoder = ATDecoder(128,32)).to(device)
        #model = GAE(encoder = VGE(data_tmp.x.shape[1],512,128), decoder = ATDecoder(128,8)).to(device)

    data_tmp = data_tmp.to(device)
    edge_index = data_tmp.edge_index.clone()

    #data_tmp = data_tmp.subgraph(data_tmp.edge_index.flatten().unique())
    data_tmp.train_mask = data_tmp.val_mask = data_tmp.test_mask = None

    data_tmp.edge_label = idx

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    final_test_acc = 0
    loss = []
    test_auc = []
    test_ap = []
    for epoch in range(1, epochs + 1):
        loss.append(edge_train(model, data_tmp, optimizer).item())
        #t_acc, te_acc = edge_test(model, data_tmp)
        #test_auc.append(t_acc)
        #test_ap.append(te_acc)
        #if te_acc > final_test_acc:
        #    final_test_acc = te_acc
    #if verbose:
    #    print("after {} epochs' training, the best test auc for edge prediction is {}".format(epochs, final_test_acc))
    model.eval()
    z = model.encode(data.x.to(device),edge_index[:,data_tmp.edge_label]) +  model.encode(data.x.to(device),edge_index[:,~data_tmp.edge_label])
    E = torch.squeeze(model.decoder.forward_all(z))
    #sm = nn.Softmax(dim=1)
    #E = sm(E)
    S = to_dense_adj(edge_index)
    if n_neighbor == 'og':
        trueE = torch.sum(S,dim=1) / S.shape[1]
        e0 = torch.quantile(E,1-2*torch.squeeze(trueE))
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    else:
        trueE = n_neighbor / data_tmp.x.shape[0]
        e0 = torch.quantile(E,1-trueE,dim=1)
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    return edge_index_new

def gps_linkres(data,model=None,lr=5e-4,weight_decay=1e-2,l=0,epochs=500,n_neighbor='og',seed=42,device='cpu',verbose=True):
    torch.manual_seed(seed)
    idx = (torch.rand(data.edge_index.shape[1])>0.5).to(device)
    data_tmp = data.clone()

    ### For large datasets (Pubmed), use 8-head attention instead of 32
    if model is None:
        #model = GAE(encoder = VGE(data.x.shape[1],512,128), decoder = ATDecoder(128,32)).to(device)
        model = GAE(encoder = VGE(data_tmp.x.shape[1],512,128), decoder = ATDecoder(128,4)).to(device)

    data_tmp = data_tmp.to(device)
    edge_index = data_tmp.edge_index.clone()

    #data_tmp = data_tmp.subgraph(data_tmp.edge_index.flatten().unique())
    data_tmp.train_mask = data_tmp.val_mask = data_tmp.test_mask = None

    data_tmp.edge_label = idx

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    final_test_acc = 0
    loss = []
    test_auc = []
    test_ap = []
    for epoch in range(1, epochs + 1):
        loss.append(edge_train(model, data_tmp, optimizer).item())
        #t_acc, te_acc = edge_test(model, data_tmp)
        #test_auc.append(t_acc)
        #test_ap.append(te_acc)
        #if te_acc > final_test_acc:
        #    final_test_acc = te_acc
    #if verbose:
    #    print("after {} epochs' training, the best test auc for edge prediction is {}".format(epochs, final_test_acc))
    model.eval()
    z = model.encode(data.x.to(device),edge_index[:,data_tmp.edge_label]) +  model.encode(data.x.to(device),edge_index[:,~data_tmp.edge_label])
    E = torch.squeeze(model.decoder.forward_all(z))
    #sm = nn.Softmax(dim=1)
    #E = sm(E)
    S = to_dense_adj(edge_index).squeeze()
    E = (1-l)*E + l*E*S
    if n_neighbor == 'og':
        trueE = torch.sum(S,dim=1) / S.shape[1]
        e0 = torch.quantile(E,1-2*torch.squeeze(trueE))
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    else:
        trueE = n_neighbor / data_tmp.x.shape[0]
        e0 = torch.quantile(E,1-trueE,dim=1)
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    return edge_index_new

def gps_regres(data,model=None, randomized=False, lr=5e-4,weight_decay=1e-2,l=0,epochs=500,n_neighbor='og',seed=42,device='cpu',verbose=True):
    torch.manual_seed(seed)
    if randomized:
        idx = (torch.rand(data.edge_index.shape[1])>0.5)
        A1 = to_dense_adj(data.edge_index[:,idx].cpu(),max_num_nodes=data.x.shape[0]).squeeze()
        A1 = torch.eye(A1.shape[0]).to(A1.device) + A1
        A2 = to_dense_adj(data.edge_index[:,~idx].cpu(),max_num_nodes=data.x.shape[0]).squeeze()
        A2 = torch.eye(A2.shape[0]).to(A2.device) + A2
        D1 = torch.squeeze(torch.sum(A1,dim=1))
        D1 = 1 / torch.sqrt(D1)
        A1_ = D1[:,None] * A1 * D1[None,:]
        A1_ = (1 / torch.sum(A1_,dim=1))[:,None] * A1_
        D2 = torch.squeeze(torch.sum(A2,dim=1))
        D2 = (1 / torch.sqrt(D2))
        A2_ = D2[:,None] * A2 * D2[None,:]
        A2_ = (1 / torch.sum(A2_,dim=1))[:,None] * A2_
        A1_csr = csr_matrix(A1_.T.numpy())
        A2_csr = csr_matrix(A2_.T.numpy())
        svd = TruncatedSVD(n_components=10, random_state=seed)
        y2 = torch.from_numpy(svd.fit_transform(A1_csr))
        y1 = torch.from_numpy(svd.fit_transform(A2_csr))
        y1 = y1.to(device)
        y2 = y2.to(device)


    else:
        idx = (torch.rand(data.edge_index.shape[1])>0.5)
        A1 = to_dense_adj(data.edge_index[:,idx].cpu(),max_num_nodes=data.x.shape[0]).squeeze()
        A1 = torch.eye(A1.shape[0]).to(A1.device) + A1
        A2 = to_dense_adj(data.edge_index[:,~idx].cpu(),max_num_nodes=data.x.shape[0]).squeeze()
        A2 = torch.eye(A2.shape[0]).to(A2.device) + A2
        D1 = torch.squeeze(torch.sum(A1,dim=1))
        D1 = 1 / torch.sqrt(D1)
        A1_ = D1[:,None] * A1 * D1[None,:]
        A1_ = (1 / torch.sum(A1_,dim=1))[:,None] * A1_
        D2 = torch.squeeze(torch.sum(A2,dim=1))
        D2 = (1 / torch.sqrt(D2))
        A2_ = D2[:,None] * A2 * D2[None,:]
        A2_ = (1 / torch.sum(A2_,dim=1))[:,None] * A2_

        U1, S1, Vh1 = torch.linalg.svd(A1_.T)
        U2, S2, Vh2 = torch.linalg.svd(A2_.T)
        y2 = torch.diag(S1[0:10]) @ Vh1[0:10,:]
        y1 = torch.diag(S2[0:10]) @ Vh2[0:10,:]
        y1 = y1.T
        y2 = y2.T
        y1 = y1.to(device)
        y2 = y2.to(device)

    data_tmp = data.clone()

    ### For large datasets (Pubmed), use 8-head attention instead of 32
    if model is None:
        model = VGE(data.x.shape[1],512,10).to(device)
        #model = GAE(encoder = VGE(data_tmp.x.shape[1],512,128), decoder = ATDecoder(128,8)).to(device)

    data_tmp = data_tmp.to(device)
    edge_index = data_tmp.edge_index.clone()

    #data_tmp = data_tmp.subgraph(data_tmp.edge_index.flatten().unique())
    data_tmp.train_mask = data_tmp.val_mask = data_tmp.test_mask = None

    data_tmp.edge_label = idx.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    final_test_acc = 0
    loss = []
    test_auc = []
    test_ap = []
    for epoch in range(1, epochs + 1):
        loss.append(node_train(model, data_tmp, y1, y2, optimizer).item())

    model.eval()
    z1 = model(data.x.to(device),edge_index[:,data_tmp.edge_label])
    z2 = model(data.x.to(device),edge_index[:,~data_tmp.edge_label])
    E = ((z1/torch.sum(z1,dim=0)) @ z1.T) + ((z2/torch.sum(z2,dim=0)) @ z2.T)
    sm = nn.Softmax(dim=1)
    E = sm(E)
    S = to_dense_adj(edge_index).squeeze()
    E = (1-l)*E + l*(E*S)
    if n_neighbor == 'og':
        trueE = torch.sum(S,dim=1) / S.shape[1]
        e0 = torch.quantile(E,1-2*torch.squeeze(trueE))
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    else:
        trueE = n_neighbor / data_tmp.x.shape[0]
        e0 = torch.quantile(E,1-trueE,dim=1)
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()



    return edge_index_new


def rs_linkprediction(data,model=None,lr=5e-4,weight_decay=1e-2,epochs=500,n_neighbor='og',seed=42,device='cpu',verbose=True):
    torch.manual_seed(seed)
    idx = (torch.rand(data.edge_index.shape[1])>0.5).to(device)
    data_tmp = data.clone()

    ### For large datasets (Pubmed), use 8-head attention instead of 32
    if model is None:
        model = GAE(encoder = LinearGE(data.x.shape[1],512,128)).to(device)
        #model = GAE(encoder = VGE(data_tmp.x.shape[1],512,128), decoder = ATDecoder(128,8)).to(device)

    data_tmp = data_tmp.to(device)
    edge_index = data_tmp.edge_index.clone()

    #data_tmp = data_tmp.subgraph(data_tmp.edge_index.flatten().unique())
    data_tmp.train_mask = data_tmp.val_mask = data_tmp.test_mask = None

    data_tmp.edge_label = idx

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    final_test_acc = 0
    loss = []
    test_auc = []
    test_ap = []
    for epoch in range(1, epochs + 1):
        loss.append(edge_train(model, data_tmp, optimizer).item())
        #t_acc, te_acc = edge_test(model, data_tmp)
        #test_auc.append(t_acc)
        #test_ap.append(te_acc)
        #if te_acc > final_test_acc:
        #    final_test_acc = te_acc
    #if verbose:
    #    print("after {} epochs' training, the best test auc for edge prediction is {}".format(epochs, final_test_acc))
    model.eval()
    z = model.encode(data.x.to(device),edge_index)
    E = torch.squeeze(model.decoder.forward_all(z))
    #sm = nn.Softmax(dim=1)
    #E = sm(E)
    S = to_dense_adj(edge_index)
    if n_neighbor == 'og':
        trueE = torch.sum(S,dim=1) / S.shape[1]
        e0 = torch.quantile(E,1-2*torch.squeeze(trueE))
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    else:
        trueE = n_neighbor / data_tmp.x.shape[0]
        e0 = torch.quantile(E,1-trueE,dim=1) - 1e-5
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    return edge_index_new

def gps_regression(data,model=None, randomized=False, lr=5e-4,weight_decay=1e-2,epochs=500,n_neighbor='og',seed=42,device='cpu',verbose=True):
    torch.manual_seed(seed)
    if randomized:
        idx = (torch.rand(data.edge_index.shape[1])>0.5)
        A1 = to_dense_adj(data.edge_index[:,idx].cpu(),max_num_nodes=data.x.shape[0]).squeeze()
        A1 = torch.eye(A1.shape[0]).to(A1.device) + A1
        A2 = to_dense_adj(data.edge_index[:,~idx].cpu(),max_num_nodes=data.x.shape[0]).squeeze()
        A2 = torch.eye(A2.shape[0]).to(A2.device) + A2
        D1 = torch.squeeze(torch.sum(A1,dim=1))
        D1 = 1 / torch.sqrt(D1)
        A1_ = D1[:,None] * A1 * D1[None,:]
        A1_ = (1 / torch.sum(A1_,dim=1))[:,None] * A1_
        D2 = torch.squeeze(torch.sum(A2,dim=1))
        D2 = (1 / torch.sqrt(D2))
        A2_ = D2[:,None] * A2 * D2[None,:]
        A2_ = (1 / torch.sum(A2_,dim=1))[:,None] * A2_
        A1_csr = csr_matrix(A1_.T.numpy())
        A2_csr = csr_matrix(A2_.T.numpy())
        svd = TruncatedSVD(n_components=10, random_state=seed)
        y2 = torch.from_numpy(svd.fit_transform(A1_csr))
        y1 = torch.from_numpy(svd.fit_transform(A2_csr))
        y1 = y1.to(device)
        y2 = y2.to(device)


    else:
        idx = (torch.rand(data.edge_index.shape[1])>0.5)
        A1 = to_dense_adj(data.edge_index[:,idx].cpu(),max_num_nodes=data.x.shape[0]).squeeze()
        A1 = torch.eye(A1.shape[0]).to(A1.device) + A1
        A2 = to_dense_adj(data.edge_index[:,~idx].cpu(),max_num_nodes=data.x.shape[0]).squeeze()
        A2 = torch.eye(A2.shape[0]).to(A2.device) + A2
        D1 = torch.squeeze(torch.sum(A1,dim=1))
        D1 = 1 / torch.sqrt(D1)
        A1_ = D1[:,None] * A1 * D1[None,:]
        A1_ = (1 / torch.sum(A1_,dim=1))[:,None] * A1_
        D2 = torch.squeeze(torch.sum(A2,dim=1))
        D2 = (1 / torch.sqrt(D2))
        A2_ = D2[:,None] * A2 * D2[None,:]
        A2_ = (1 / torch.sum(A2_,dim=1))[:,None] * A2_

        U1, S1, Vh1 = torch.linalg.svd(A1_.T)
        U2, S2, Vh2 = torch.linalg.svd(A2_.T)
        y2 = torch.diag(S1[0:10]) @ Vh1[0:10,:]
        y1 = torch.diag(S2[0:10]) @ Vh2[0:10,:]
        y1 = y1.T
        y2 = y2.T
        y1 = y1.to(device)
        y2 = y2.to(device)

    data_tmp = data.clone()

    ### For large datasets (Pubmed), use 8-head attention instead of 32
    if model is None:
        model = VGE(data.x.shape[1],512,10).to(device)
        #model = GAE(encoder = VGE(data_tmp.x.shape[1],512,128), decoder = ATDecoder(128,8)).to(device)

    data_tmp = data_tmp.to(device)
    edge_index = data_tmp.edge_index.clone()

    #data_tmp = data_tmp.subgraph(data_tmp.edge_index.flatten().unique())
    data_tmp.train_mask = data_tmp.val_mask = data_tmp.test_mask = None

    data_tmp.edge_label = idx.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    final_test_acc = 0
    loss = []
    test_auc = []
    test_ap = []
    for epoch in range(1, epochs + 1):
        loss.append(node_train(model, data_tmp, y1, y2, optimizer).item())

    model.eval()
    z1 = model(data.x.to(device),edge_index[:,data_tmp.edge_label])
    z2 = model(data.x.to(device),edge_index[:,~data_tmp.edge_label])
    E = ((z1/torch.sum(z1,dim=0)) @ z1.T) + ((z2/torch.sum(z2,dim=0)) @ z2.T)
    sm = nn.Softmax(dim=1)
    E = sm(E)
    S = to_dense_adj(edge_index)
    if n_neighbor == 'og':
        trueE = torch.sum(S,dim=1) / S.shape[1]
        e0 = torch.quantile(E,1-2*torch.squeeze(trueE))
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    else:
        trueE = n_neighbor / data_tmp.x.shape[0]
        e0 = torch.quantile(E,1-trueE,dim=1)
        relu = nn.ReLU()
        edge_index_new = relu(E-e0[:,None]).nonzero().t().contiguous()

    return edge_index_new
