import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import copy
import scipy.io as sio


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def label_propagation(adj, labels, idx, K, alpha): 
    y0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    for i in idx:
        y0[i][labels[i]] = 1.0
    
    y = y0
    for i in range(K): 
        y = torch.matmul(adj, y)
        for i in idx:
            y[i] = F.one_hot(torch.tensor(labels[i].cpu().numpy().astype(np.int64)), labels.max().item() + 1)
        y = (1 - alpha) * y + alpha * y0

    return y


def random_planetoid_splits_original(labels, num_classes, percls_trn=20, val_lb=500, test_num=0, Flag=1):

    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                           for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_idx = train_index
    val_idx = val_index
    test_idx = rest_index

    return train_idx, val_idx, test_idx


def load_data(dataset_source, train_rate=20, val_rate=30):
    data = sio.loadmat("data/{}.mat".format(dataset_source))
    features = data["Attributes"]
    if dataset_source in ['physics','photo']:
        features = features.todense()
    adj = data["Network"]
    labels = data["Label"][0]


    nb_class = max(labels) + 1
    num_classes = nb_class
    N = len(labels)
    
    labels = torch.LongTensor(labels)

    test_num =  N-int(train_rate)*num_classes-int(val_rate)*num_classes

    idx_train, idx_val, idx_test = random_planetoid_splits_original(labels, num_classes, int(train_rate), int(val_rate), test_num)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if not dataset_source in ['photo']:
        adj_sp = normalize_adj(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features))
    adj = sparse_mx_to_torch_sparse_tensor(adj_sp)
    labels = torch.LongTensor(labels.numpy().reshape((1,-1))[0])

    return adj, features, labels, idx_train, idx_val, idx_test, adj_sp


def graph_augmentation(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    adj_pred = adj_pred + adj_pred.T

    return adj_pred