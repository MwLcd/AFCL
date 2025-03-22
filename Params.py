import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=2048, type=int, help='batch size')
    parser.add_argument('--tstBat', default=1024, type=int, help='number of users in a testing batch')
    parser.add_argument('--reg', default=1e-4, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=400, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=64, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
    parser.add_argument('--topk10', default=10, type=int, help='K of top K')
    parser.add_argument('--topk20', default=20, type=int, help='K of top K')
    parser.add_argument('--topk40', default=40, type=int, help='K of top K')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--ssl_reg', default=0.1, type=float, help='weight for contrative learning')
    parser.add_argument("--ib_reg", type=float, default=0.1, help='weight for information bottleneck')
    parser.add_argument('--temp', default=0.4, type=float, help='temperature in contrastive learning')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default=1, type=int, help='indicates which gpu to use')
    parser.add_argument("--seed", type=int, default=421, help="random seed")
    parser.add_argument('--model_name', default='mine', type=str, help='the name of the model')


args = ParseArgs()