import argparse

# Training settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora_geometric',
                        help='Choose from {cora_geometric, citeseer_geometric, pubmed, cs, physics, photo}')
    parser.add_argument('--K', type=int, default=10,
                        help='the depth of appnp and ptt when training')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='the alpha of appnp and ptt when training')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='Set weight decay.')
    parser.add_argument('--loss_decay', type=float, default=0.05,
                        help='Set loss_decay.')
    parser.add_argument('--fast_mode', type=bool, default=False,
                        help='whether propogate when validation.')
    parser.add_argument('--epsilon', type=int, default=100,
                        help='Set importance change of f(x).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--seed', type=int, default=2144199737, help='Random seed for split data.')
    parser.add_argument('--ini_seed', type=int, default=2144199730, help='Random seed to initialize parameters.')
    parser.add_argument('--N', type=int, default=5, help='number of training samples')
    parser.add_argument('--step', type=int, default=3, help='The steps for graph topology augmentation.')
    parser.add_argument('--remove_pct', type=int, default=2,
                        help='percentage of edges to remove in augmentation.')
    parser.add_argument('--add_pct', type=int, default=2,
                        help='percentage of edges to add in augmentation.')
    return parser.parse_args()
