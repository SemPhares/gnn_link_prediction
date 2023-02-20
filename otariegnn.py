import dgl
import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv
from dgl.nn import SAGEConv
import dgl.function as fn

import itertools
# import scipy.sparse as sp
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class GCN(torch.nn.Module):
    def __init__(self, 
                 in_feats, 
                 h_feats,
                 out_dim,
                 n_layers, 
                 activation):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        
         # input layer
        self.layers.append(GraphConv(in_feats, h_feats, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(h_feats, h_feats, activation=activation))
        # output layer
        self.layers.append(GraphConv(h_feats, out_dim, activation=None))
        
    def forward(self, g, in_feat):
        h = in_feat
        for layer in self.layers:
            h = layer(g, h)
        return h
    

class GraphSAGEModel(torch.nn .Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGEModel, self).__init__()
        self.layers = torch.nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type,
                                         feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type,
                                             feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, out_dim, aggregator_type,
                                         feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h 

class DotPredictor(torch.nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]
            
            
def train_test_split(g, test_ratio=0.2):
    """
    Split the g graph in train and test set
    
    @g : the model to split
    @test_ratio : the ratio of spliting
    
    """
    
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * test_ratio)  # number of edges in test set
    train_size = g.number_of_edges() - test_size  # number of edges in train set

    # get positive edges for test and train
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges
    # adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj = g.adj(scipy_fmt='coo')
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    # split the negative edges for training and testing 
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # construct positive and negative graphs for training and testing
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    # training graph
    train_g = dgl.remove_edges(g, eids[:test_size])
    train_g = dgl.add_self_loop(train_g)
    
    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g


def HomeMadeModel(g, model_name='SAGE',aggregator_type='mean', n_layers=2, hidden_size=16, dropout=0.5,print_model=False):
    """
    Return the model to train
    
    @model_name : can be GCN or SAGE
    @aggregator_type : aggregation function in ['mean', 'gcn', 'pool', 'lstm']
    @n_layers : number of layers of the model
    @hidden_size : the size of the hidden layer in the neural net and the output dimmension of the model
    @dropout : rate of dropout of the model
    
    """
    
    if model_name == 'GCN':
        model = GCN(g.ndata['feat'].shape[1], hidden_size, hidden_size, n_layers, F.relu )
        
    elif model_name == 'SAGE':
#         model = GraphSAGE(train_g.ndata['feat'].shape[1], hidden_size)
        model = GraphSAGEModel(g.ndata['feat'].shape[1],hidden_size,hidden_size,n_layers, F.relu, dropout, aggregator_type)
        
    if print_model==True:
        print('---------------- MODEL ----------------')
        print(model)

    return model



def train(model, g, test_ratio=0.2, n_epochs=100, l_rate=0.01,verbose=1, plot_loss=1):
    """
    train the model on dgl graph g
    
    @model : the GNN model to train
    @g : the dgl graph to train on
    @test_ratio : the ratio for train test spliting
    @n_epochs : number of epoch to train on
    @l_rate : the learning rate of the optimizer
    
    """

    pred = DotPredictor()
    
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = train_test_split(g, test_ratio=0.2)

    def compute_loss(pos_score, neg_score):  # computes the loss based on binary cross entropy
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_auc(pos_score, neg_score):  # computes AUC (Area-Under-Curve) score
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
        return roc_auc_score(labels, scores)


    # ----------- set up loss and optimizer -------------- #
    # in this case, loss will in training loop
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=l_rate)
    
     # ---------------------- train ------------------------- #
    print('---------------- TRAINING ----------------')
    all_loss = []
    
    for e in tqdm(range(n_epochs)): #e in range(n_epochs):
        # forward
        h = model(train_g, train_g.ndata['feat'])  # get node embeddings
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        all_loss.append(loss)
        optimizer.step()

        if e % 5 == 0 and verbose ==1:
            print('Epoch {}, loss: {}'.format(e, loss))

    # ----------- test and check results ---------------- #
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        auc =  compute_auc(pos_score, neg_score)
        
        print('\n','---------------- AUC ----------------')
        print('AUC', auc)
    
    # ----------- plot results ---------------- #
    if plot_loss == 1:
        
        print('\n','---------------- RESULTS ----------------')
        plt.plot(all_loss)
        plt.title('Loss during training')
        plt.text(n_epochs-25,max(all_loss),f'AUC: {round(auc,3)}',c='b')
        plt.show()
    
    return h , auc


def train_grid(g,verbose=1, auto=True, **params):
    """
    Train the composed model on a grid to search the best parameters
    
    @g : the dgl graph to train on
    @verbose : verbosity
    @auto : etheir train on a pre-setted grid or not 
    @params a dict of paramaters to train on:
        if auto = true :
            params = {'model_name' : ['GCN','SAGE'],
                  'aggregator_type' : ['mean', 'gcn', 'lstm'],    #'pool', 
                  'n_layers' : [int(_) for _ in np.linspace(2,15,5)],
                  'hidden_size' : [int(_) for _ in np.linspace(3,30,5)],
                  'n_epochs' : [int(_) for _ in np.linspace(100,2000,5)],
                  'l_rate' : [1e-4,1e-3,1e-2,1e-1]} 
                  
    """
    grid_results = {}
    
    if auto == True:
        params = {'model_name' : ['GCN','SAGE'],
                  'aggregator_type' : ['mean', 'gcn', 'lstm'],    #'pool', 
                  'n_layers' : [int(_) for _ in np.linspace(2,15,5)],
                  'hidden_size' : [int(_) for _ in np.linspace(3,30,5)],
                  'n_epochs' : [int(_) for _ in np.linspace(100,2000,5)],
                  'l_rate' : [1e-4,1e-3,1e-2,1e-1]}
        
    else:
        params = params
        
        param_list =  [ 'model_name',
                        'aggregator_type',
                        'n_layers',
                        'hidden_size',
                        'n_epochs',
                        'l_rate' ]
        
        if not 'model_name' in params:
            params['model_name'] = ['SAGE']
        
        if not 'aggregator_type' in params:
            params['aggregator_type'] = ['mean']        
        
        if not 'n_layers' in params:
            params['n_layers'] = [2]  
            
        if not 'hidden_size' in params:
            params['hidden_size'] = [16]
            
        if not 'n_epochs' in params:
            params['n_epochs'] = [100]
            
        if not 'l_rate' in params:
            params['l_rate'] = [1e-2]
            
            
    assert all(_ in params for _ in param_list) , "Please fill the params dict"
    
    for model_name in params['model_name']:
        for n_layers in params['n_layers']:
            for hidden_size in params['hidden_size']:
                for n_epochs in params['n_epochs']:
                    for l_rate in params['l_rate']:
                        
                        if model_name == 'GCN':
                            model = HomeMadeModel(g, model_name=model_name,aggregator_type=None ,n_layers=n_layers, hidden_size=hidden_size,print_model=False)
                            _ , auc = train(model,g,test_ratio=0.2, n_epochs=n_epochs, l_rate=l_rate, verbose=0, plot_loss=0)
                            grid_results[f'model_name: {model_name} , aggregator_type: {None}, n_layers: {n_layers} ,hidden_size: {hidden_size} ,n_epochs: {n_epochs} ,l_rate: {l_rate}'] = auc
                            
                            if verbose == 1:
                                print(f'model_name: {model_name}, n_layers: {n_layers} ,hidden_size: {hidden_size} ,n_epochs: {n_epochs} ,l_rate: {l_rate}')
                                print('AUC :',auc)
                                            
                        else:
                            for aggregator_type in params['aggregator_type']:
                                model = HomeMadeModel(g, model_name=model_name,aggregator_type=aggregator_type ,n_layers=n_layers, hidden_size=hidden_size,print_model=False)
                                _ , auc = train(model,g,test_ratio=0.2, n_epochs=n_epochs, l_rate=l_rate, verbose=0, plot_loss=0)
                                grid_results[f'model_name: {model_name} , aggregator_type: {aggregator_type}, n_layers: {n_layers} ,hidden_size: {hidden_size} ,n_epochs: {n_epochs} ,l_rate: {l_rate}'] = auc
                                
                                if verbose == 1:
                                    print(f'model_name: {model_name} ,aggregator_type: {aggregator_type} , n_layers: {n_layers} ,hidden_size: {hidden_size} ,n_epochs: {n_epochs} ,l_rate: {l_rate}')
                                    print('AUC :',auc)
                                
    grid_results = sorted(grid_results.items(), key=lambda x: -x[1])
    best_ = next(iter(grid_results))
                            
    print(best_)
                            
    return grid_results, best_
                            
    
    
def GenerateLinkScores(h, g, user_id=0, nb_links=10):
    """
    @h : the node embeddings, with shape [num_nodes, hidden_size]
    @g : DGL graph
    @user_id : the identifier of the user for whom the links are to be predicted
    @nb_links : the number of most important links to print @
    
    # one end of the edge is user_id
    # the other end is a user that's NOT friends with user_id
    
    """
    
    u, v = g.edges()
    number_of_nodes = g.number_of_nodes()
    
    user_friends = set()
    user_neg_u, user_neg_v = [], []
    for n1, n2 in zip(u, v):   # get all friends of user_id
        if int(n1) == user_id:
            user_friends.add(int(n2))
        if int(n2) == user_id:
            user_friends.add(int(n1))

    for i in range(number_of_nodes):  # generate "negative edges" for user_id
        if i != user_id and i not in user_friends:
            user_neg_u.append(user_id)
            user_neg_v.append(i)
    
    #generate a graph with (num_nodes - num_friends_of_user) edges
    user_neg_g = dgl.graph((user_neg_u, user_neg_v), num_nodes=number_of_nodes)
    
    pred = DotPredictor()

    # calculate the score of each user
    pos_scores = [(i, score, score.sigmoid()) for i, score in enumerate(pred(g, h)) if i in user_friends]
    neg_scores = [(i, score, score.sigmoid()) for i, score in enumerate(pred(user_neg_g, h))]
    
    
    # produce final ranked list
    pos_scores.sort(key=lambda x: -x[1])
    neg_scores.sort(key=lambda x: -x[1])
    
    #------------------------------------- HYPOTHESE DANGEREUSE 
    max_pos_score = pos_scores[-1][1].detach()
    scores = [elem for elem in neg_scores if elem[1]> max_pos_score]

    # display results
    print(f"List of 5 suggested friends for user {user_id}:")
    for i in range(nb_links):
        print(f'- User {neg_scores[i][0]}, score = {neg_scores[i][1]}, proba = {neg_scores[i][2]}')
    
    return scores


