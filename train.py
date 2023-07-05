import arguments
import numpy as np

import torch
import torch.nn.functional as F

from utils import *
from models import AGST
import random
from early_stop import EarlyStopping, Stop_args
from sklearn.preprocessing import normalize

from copy import deepcopy


args = arguments.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.ini_seed)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(args.ini_seed)





def main(adj, features, labels, idx_train, idx_val, idx_test, step):

    # Label Propagation, Teacher
    y_soft_train = label_propagation(adj, labels, idx_train, args.K, args.alpha)

    features = features.to(device)
    adj = adj.to(device)
    y_soft_train = y_soft_train.to(device)
    labels = labels.to(device)

    idx_all = list(range(len(labels)))
    idx_warm = list(set(idx_all).difference(set(idx_train.tolist())))
    random.shuffle(idx_warm)

    nclass = labels.max().item() + 1

    agst_model = AGST(best_state=None,
                    nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=nclass,
                    dropout=args.dropout,
                    epsilon=args.epsilon,
                    K=args.K,
                    alpha=args.alpha,
                    nshot=args.N,
                    m=0.99).to(device)

    agst_optimizer = torch.optim.Adam(agst_model.parameters(), lr=0.1)

    def train_constrastic(epoch):
        agst_model.train()

        output, query_features, prototype = agst_model(features, idx_train)

        pre_label = agst_model.model.inference(output, adj).max(1)[1]

        label_predict = torch.Tensor(pre_label[idx_warm].cpu().numpy()).unsqueeze(1).to(torch.int64).to(device)

        proto_norm_momt = torch.Tensor(np.array([0.1]*nclass)).to(device)

        proto_loss = agst_model.get_proto_loss(query_features[idx_warm], label_predict, prototype, proto_norm_momt)

        # classification loss for labeled nodes
        loss_train = args.loss_decay * (F.nll_loss(output[idx_train], labels[idx_train]))
        # classification loss for ynlabeled nodes
        loss_train += args.loss_decay * (agst_model.model.loss_function(y_hat=output[idx_warm], y_soft=y_soft_train[idx_warm], epoch=epoch))
        # contrastic loss
        loss_train += 0.1 * proto_loss
        # regularization loss
        loss_train += args.weight_decay * torch.sum(agst_model.model.encoder.weight ** 2) / 2

        # Update the model
        agst_optimizer.zero_grad()
        loss_train.backward()
        agst_optimizer.step()

        acc_train = accuracy(output[idx_train], labels[idx_train])

        model = copy.deepcopy(agst_model.model)
        model.eval()
        _, output = model(features)

        if not args.fast_mode:
            output = model.inference(output, adj)
        acc_val = accuracy(output[idx_val], labels[idx_val])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))

        return acc_val.item()

    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(agst_model, **stopping_args)
    for epoch in range(args.epochs):
        acc_val = train_constrastic(epoch)
        if early_stopping.check([acc_val], epoch):
            break

    # Restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    agst_model.load_state_dict(early_stopping.best_state)

    # test
    agst_model.eval()
    _, output = agst_model.model(features)

    output = agst_model.model.inference(output, adj)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_val = accuracy(output[idx_val], labels[idx_val])


    return acc_test.item(), acc_val.item(), agst_model


if __name__ == "__main__":
    # Load data and pre_process data
    adj, features, labels, idx_train, idx_val, idx_test, adj_sp = load_data(args.dataset, args.N)

    best_val_acc = 0
    best_test_acc = 0
    adj_pred = adj
    for i in range(args.step):
        acc_test, acc_val, agst_model = main(adj_pred, features, labels, idx_train, idx_val, idx_test, i)

        if acc_val >= best_val_acc:
            best_test_acc = acc_test
            best_val_acc = acc_val
        # Graph Topology Augmentation
        features = features.to(device)
        adj = adj.to(device)
        A_pred = agst_model.model.get_edge_prob(features, adj).detach().cpu().numpy()

        adj_pred = graph_augmentation(adj_sp, A_pred, args.remove_pct, args.add_pct)

        adj_pred = normalize_adj(adj_pred)
        adj_pred = sparse_mx_to_torch_sparse_tensor(adj_pred)

    print()
    print("The best test result,",
          "accuracy= {:.4f}".format(best_test_acc))