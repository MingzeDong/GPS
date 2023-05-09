from torch_geometric.utils import dropout_adj, negative_sampling
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from ray import tune
from . import gps, digl, sdrf,sdrf_gpu, train
from torch_geometric.data import Data
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from torch_geometric.nn import knn_graph

def rayGCN1train(config,data,device='cpu',raytune=False,seed = 42):
    torch.manual_seed(seed)
    epochs=200

    edge_index = data.edge_index
    model = train.GCN_1layer(data.x.shape[1], int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc

def rayGCNtrain(config,data,device='cpu',raytune=False,seed = 42):
    torch.manual_seed(seed)
    epochs=200

    edge_index = data.edge_index
    model = train.GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc


def rayKNNtrain(config,data,device='cpu',raytune=False,seed = 42, only_edge_index = False):
    torch.manual_seed(seed)
    epochs=200
    x_csr = csr_matrix(data.x.to("cpu").numpy())
    svd = TruncatedSVD(n_components=10, random_state=seed)
    data_svd = svd.fit_transform(x_csr)
    edge_index = knn_graph(torch.from_numpy(data_svd),k=config["n"])

    if only_edge_index:
        return Data(x=data.x.clone(),edge_index = edge_index,y = data.y.clone())

    model = train.GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc


def rayDIGLtrain(config,data,device='cpu',raytune=False,seed = 42, only_edge_index = False):
    torch.manual_seed(seed)
    epochs=200

    edge_index = digl.gdc(data.edge_index,alpha=config["alpha"], eps=config["eps"])
    if only_edge_index:
        return Data(x=data.x.clone(),edge_index = edge_index,y = data.y.clone())
    model = train.GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc



def raySDRFtrain(config,data,device='cpu',raytune=False,seed = 42, only_edge_index = False):
    torch.manual_seed(seed)
    epochs=200
    if device=='cpu':
        data_sdrf = sdrf.sdrf(data,loops=config["loops"],tau=config["tau"],removal_bound=config["removal_bound"],is_undirected=False)
    else:
        data_sdrf = sdrf_gpu.sdrf(data,loops=config["loops"],tau=config["tau"],removal_bound=config["removal_bound"],is_undirected=False)

    edge_index = data_sdrf.edge_index
    if only_edge_index:
        return Data(x=data.x.clone(),edge_index =edge_index,y = data.y.clone())

    model = train.GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc



def rayGPStrain(config,data,device='cpu',raytune=False,seed = 42, only_edge_index = False,verbose=False):
    torch.manual_seed(seed)
    epochs=200

    edge_index = gps.gps_linkprediction(data,model=None,lr=config["lr_e"], weight_decay=config["weight_decay_e"],epochs=500,seed=seed,n_neighbor=config["n"],device=device,verbose=verbose)
    if only_edge_index:
        return Data(x=data.x.clone(),edge_index = edge_index,y = data.y.clone())

    model = train.GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc
    
def rayGPSrestrain(config,data,device='cpu',raytune=False,seed = 42, only_edge_index = False,verbose=False):
    torch.manual_seed(seed)
    epochs=200

    edge_index = gps.gps_linkres(data,model=None,lr=config["lr_e"], weight_decay=config["weight_decay_e"],l=config["l"],epochs=500,seed=seed,n_neighbor=config["n"],device=device,verbose=verbose)
    if only_edge_index:
        return Data(x=data.x.clone(),edge_index = edge_index,y = data.y.clone())

    model = train.GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc
    

def rayRStrain(config,data,device='cpu',raytune=False,seed = 42, only_edge_index = False):
    torch.manual_seed(seed)
    epochs=200

    edge_index = gps.rs_linkprediction(data,model=None,lr=config["lr_e"], weight_decay=config["weight_decay_e"],epochs=500,seed=seed,n_neighbor=config["n"],device=device,verbose=False)
    if only_edge_index:
        return Data(x=data.x.clone(),edge_index = edge_index,y = data.y.clone())

    model = train.GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc

def rayGPSRtrain(config,data,randomized=False,device='cpu',raytune=False,seed = 42, only_edge_index = False,verbose = False):
    torch.manual_seed(seed)
    epochs=200

    edge_index = gps.gps_regression(data,model=None,randomized=randomized, lr=config["lr_e"], weight_decay=config["weight_decay_e"],epochs=500,seed=seed,n_neighbor=config["n"],device=device,verbose=verbose)
    if only_edge_index:
        return Data(x=data.x.clone(),edge_index = edge_index,y = data.y.clone())

    model = train.GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc


def rayGPSRrestrain(config,data,randomized=False,device='cpu',raytune=False,seed = 42, only_edge_index = False, verbose = False):
    torch.manual_seed(seed)
    epochs=200

    edge_index = gps.gps_regres(data,model=None,randomized=randomized,lr=config["lr_e"], weight_decay=config["weight_decay_e"],l=config["l"],epochs=500,seed=seed,n_neighbor=config["n"],device=device,verbose=verbose)
    if only_edge_index:
        return Data(x=data.x.clone(),edge_index = edge_index,y = data.y.clone())

    model = train.GCN(data.x.shape[1], 100, int(torch.max(data.y)+1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    edge_index = edge_index.to(device)

    best_val_acc = final_test_acc = 0
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss.append(train.train(model, data, edge_index, optimizer, loss_fn).item())
        t_acc, v_acc, te_acc = train.test(model, data, edge_index)
        train_acc.append(t_acc.cpu())
        val_acc.append(v_acc.cpu())
        test_acc.append(te_acc.cpu())
        if raytune:
            tune.report(mean_accuracy=float(v_acc.cpu().detach().numpy()))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            final_test_acc = te_acc
    if raytune == False:
        return train_acc,val_acc,test_acc, final_test_acc
