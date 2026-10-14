import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_sparse import SparseTensor
from utils import row_normlize_sparsetensor
from torch_sparse import spmm as sparsespmm
from torch.nn import Conv1d, MaxPool1d, Linear

# import dgl.function as fn
# import dgl
from torch_geometric.graphgym import AtomEncoder, BondEncoder
from utils import compute_identity

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels, multi_class):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, class_channels)
        self.multi_class = multi_class

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.0, training=self.training) # default is 0.5
        x = self.conv2(x, edge_index)
        if self.multi_class:
            return x
        else:
            return F.sigmoid(x)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, class_channels, multi_class):
        super().__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.5)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, class_channels, heads=1, concat=False, dropout=0.5)
        self.multi_class = multi_class

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        if self.multi_class:
            return x
        else:
            return F.sigmoid(x)


class SAGE_sup(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels, num_layers, multi_class):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, class_channels))
        self.multi_class = multi_class
        self.num_layers = 2

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        if self.multi_class:
            return x
        else:
            return F.sigmoid(x)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id]
                x = conv(x, batch.edge_index)
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        if self.multi_class:
            return x
        else:
            return F.sigmoid(x)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels, multi_class):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, class_channels)
        self.multi_class = multi_class
    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        out = self.fc3(x)
        if self.multi_class:
            return x
        else:
            return F.sigmoid(x)


class H2GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, multi_class):
        super(H2GCN, self).__init__()
        # input
        self.dense1 = torch.nn.Linear(nfeat, nhid)
        # output
        self.dense2 = torch.nn.Linear(nhid*7, nclass)
        # drpout
        # self.dropout = SparseDropout(dropout)
        self.dropout = dropout
        # conv
        self.conv1 = GCNConv(nhid, nhid,
                             #cached=True, normalize=False
                             )
        self.conv2 = GCNConv(nhid*2, nhid*2,
                             #cached=True, normalize=False
                             )
        self.relu = torch.nn.ReLU()
        self.vec = torch.nn.Flatten()
        self.iden = torch.sparse.Tensor()

        self.multi_class = multi_class

    def forward(self, features, edge_index):

        # feature space ----> hidden
        # adj2 = adj * adj
        # r1: compressed feature matrix
        x = self.relu(self.dense1(features))
        # # vectorize
        # x = self.vec(x)
        # aggregate info from 1 hop away neighbor
        # r2 torch.cat(x, self.conv(x, adj), self.conv(x, adj2))
        x11 = self.conv1(x, edge_index)
        x12 = self.conv1(x11, edge_index)
        x1 = torch.cat((x11, x12), -1)

        # vectorize
        # x = self.vec(x1)
        # aggregate info from 2 hp away neighbor
        x21 = self.conv2(x1, edge_index)
        x22 = self.conv2(x21, edge_index)
        x2 = torch.cat((x21, x22), -1)

        # concat
        x = torch.cat((x, x1, x2), dim=-1)
        # x = self.dropout(x)
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)

        if self.multi_class:
            return x
        else:
            return F.sigmoid(x)



class GCN_LPA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_gcn, lpa_iter, edge_weights, multi_class):
        super().__init__()

        self.num_gcn = num_gcn
        # default: add self loop, normalize
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_gcn):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
            elif i == self.num_gcn-1:
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))

        self.lpa_iter = lpa_iter
        self.edge_attr = torch.nn.Parameter(edge_weights.abs(), requires_grad=True)

        self.multi_class = multi_class

    def forward(self, x, soft_labels, edge_index):
        weighted_adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=self.edge_attr, sparse_sizes=(x.shape[0], x.shape[0]))
        weighted_adj = row_normlize_sparsetensor(weighted_adj)
        #gcn

        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.convs[0](x, weighted_adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, weighted_adj)
        if self.multi_class:
            x = x
        else:
            m = torch.nn.Sigmoid()
            x = m(x)

        #lpa
        predicted_labels = soft_labels
        _, _, value = weighted_adj.coo()
        for i in range(self.lpa_iter):
            predicted_labels = sparsespmm(edge_index, value, predicted_labels.shape[0], predicted_labels.shape[0], predicted_labels)
        if self.multi_class:
            predicted_labels = predicted_labels
        else:
            predicted_labels = m(predicted_labels)
        
        return x, predicted_labels


class LANC(torch.nn.Module):
    def __init__(self, in_channels, class_channels, num_label, multi_class):
        super().__init__()
        self.conv1 = Conv1d(in_channels, 16, 2)
        self.conv2 = Conv1d(in_channels, 16, 3)
        self.conv3 = Conv1d(in_channels, 16, 4)
        self.conv4 = Conv1d(in_channels, 16, 5)
        self.mlp = nn.Sequential(
                                Linear(192, 64),
                                nn.ReLU(),
                                Linear(64, class_channels))
        self.mlp1 = nn.Sequential(Linear(128, 64),
                                  nn.ReLU(),
                                  Linear(64, class_channels))

        self.attention = nn.Sequential(nn.Linear(192, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 1, bias=False))
        self.lbl_emb = nn.Embedding(num_label, 128)

        self.multi_class = multi_class

    def forward(self, x, y):
        y = self.lbl_emb(y)

        x = x.permute(0, 2, 1)
        #convolution
        x1 = self.conv1(x)

        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.6)

        x2 = F.dropout(F.relu(self.conv2(x)), p=0.6)
        x3 = F.dropout(F.relu(self.conv3(x)), p=0.6)
        x4 = F.dropout(F.relu(self.conv4(x)), p=0.6)

        x1 = torch.amax(x1, 2)
        x2 = torch.amax(x2, 2)
        x3 = torch.amax(x3, 2)
        x4 = torch.amax(x4, 2)
        # feature vector
        out = torch.cat((x1, x2, x3, x4), dim=1)
        # (64, 64)

        # attention vector
        # (batch_size, node embedding + label embedding dimension)
        s = []
        for i in range(y.shape[0]):
            # 64 is the batch size
            support = torch.cat(out.shape[0] * [y[i]]).reshape(out.shape[0], -1)
            # concat the node emb with one label emb
            c = torch.hstack((out, support))
            s.append(c)
        #print(torch.stack(s).shape)
        s = self.attention(torch.stack(s))#.squeeze()
        s = s.squeeze()
        a = F.softmax(torch.transpose(s, 0, 1), dim=1)

        att_vec = torch.mm(a, y)

        # concat feature vector and attention vector
        emb = torch.cat((out, att_vec), dim=1)

        # use label embedding to predict the labels
        emb_pre = self.mlp(emb)

        # try padding in label embedding and use the same mlp as emb
        pad = torch.zeros(y.shape[0], emb.shape[1] - y.shape[1])
        y = torch.hstack((pad, y))
        y_pre = self.mlp(y)

        if self.multi_class:
            return emb_pre, y_pre
        else:
            return torch.sigmoid(emb_pre), y_pre



##################################################################### LSPE ########################################################
class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, graph_norm=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.graph_norm = graph_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
    
    def forward(self, g, h, p=None, e=None, snorm_n=None):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        # GN from benchmarking-gnns-v1
        if self.graph_norm:
            h = h * snorm_n
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, None, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)



class GatedGCNLSPELayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, use_lapeig_loss=False, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_lapeig_loss = use_lapeig_loss
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A1 = nn.Linear(input_dim*2, output_dim, bias=True)
        self.A2 = nn.Linear(input_dim*2, output_dim, bias=True)
        self.B1 = nn.Linear(input_dim, output_dim, bias=True)
        self.B2 = nn.Linear(input_dim, output_dim, bias=True)
        self.B3 = nn.Linear(input_dim, output_dim, bias=True)
        self.C1 = nn.Linear(input_dim, output_dim, bias=True)
        self.C2 = nn.Linear(input_dim, output_dim, bias=True)
        
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        # self.bn_node_p = nn.BatchNorm1d(output_dim)

    def message_func_for_vij(self, edges):
        hj = edges.src['h'] # h_j
        pj = edges.src['p'] # p_j
        vij = self.A2(torch.cat((hj, pj), -1))
        return {'v_ij': vij} 
    
    def message_func_for_pj(self, edges):
        pj = edges.src['p'] # p_j
        return {'C2_pj': self.C2(pj)}
       
    def compute_normalized_eta(self, edges):
        return {'eta_ij': edges.data['sigma_hat_eta'] / (edges.dst['sum_sigma_hat_eta'] + 1e-6)} # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
      
    def forward(self, g, h, p, e, snorm_n):   

        with g.local_scope():
        
            # for residual connection
            h_in = h 
            p_in = p 
            e_in = e 

            # For the h's
            g.ndata['h']  = h
            g.ndata['A1_h'] = self.A1(torch.cat((h, p), -1)) 
            # self.A2 being used in message_func_for_vij() function
            g.ndata['B1_h'] = self.B1(h)
            g.ndata['B2_h'] = self.B2(h) 

            # For the p's
            g.ndata['p'] = p
            g.ndata['C1_p'] = self.C1(p)
            # self.C2 being used in message_func_for_pj() function

            # For the e's
            g.edata['e']  = e 
            g.edata['B3_e'] = self.B3(e) 

            #--------------------------------------------------------------------------------------#
            # Calculation of h
            g.apply_edges(fn.u_add_v('B1_h', 'B2_h', 'B1_B2_h'))
            g.edata['hat_eta'] = g.edata['B1_B2_h'] + g.edata['B3_e']
            g.edata['sigma_hat_eta'] = torch.sigmoid(g.edata['hat_eta'])
            g.update_all(fn.copy_e('sigma_hat_eta', 'm'), fn.sum('m', 'sum_sigma_hat_eta')) # sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.compute_normalized_eta) # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.message_func_for_vij) # v_ij
            g.edata['eta_mul_v'] = g.edata['eta_ij'] * g.edata['v_ij'] # eta_ij * v_ij
            g.update_all(fn.copy_e('eta_mul_v', 'm'), fn.sum('m', 'sum_eta_v')) # sum_j eta_ij * v_ij
            g.ndata['h'] = g.ndata['A1_h'] + g.ndata['sum_eta_v']

            # Calculation of p
            g.apply_edges(self.message_func_for_pj) # p_j
            g.edata['eta_mul_p'] = g.edata['eta_ij'] * g.edata['C2_pj'] # eta_ij * C2_pj
            g.update_all(fn.copy_e('eta_mul_p', 'm'), fn.sum('m', 'sum_eta_p')) # sum_j eta_ij * C2_pj
            g.ndata['p'] = g.ndata['C1_p'] + g.ndata['sum_eta_p']

            #--------------------------------------------------------------------------------------#

            # passing towards output
            h = g.ndata['h'] 
            p = g.ndata['p']
            e = g.edata['hat_eta'] 

            # GN from benchmarking-gnns-v1
            
            #h = h * snorm_n
            
            # batch normalization  
            if self.batch_norm:
                h = self.bn_node_h(h)
                e = self.bn_node_e(e)
                # No BN for p

            # non-linear activation
            h = F.relu(h) 
            e = F.relu(e) 
            p = torch.tanh(p)

            # residual connection
            if self.residual:
                h = h_in + h 
                p = p_in + p
                e = e_in + e 

            # dropout
            h = F.dropout(h, self.dropout, training=self.training)
            p = F.dropout(p, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

            return h, p, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        input_dim = net_params.input_dim
        hidden_dim = net_params.hidden_dim
        out_dim = net_params.out_dim
        n_classes = net_params.n_classes
        dropout = net_params.dropout
        n_layers = net_params.L
        self.readout = net_params.readout
        self.batch_norm = net_params.batch_norm
        self.residual = net_params.residual
        self.device = net_params.device
        self.pe_init = net_params.pe_init
        self.n_classes = net_params.n_classes
        
        self.use_lapeig_loss = net_params.use_lapeig_loss
        self.lambda_loss = net_params.lambda_loss
        self.alpha_loss = net_params.alpha_loss
        
        self.pos_enc_dim = net_params.pos_enc_dim
        
        if self.pe_init in ['rand_walk', 'lap_pe']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)

        ##### the encoders for the OGB datasets, thus disabled for my datasets ####
        #self.atom_encoder = AtomEncoder(hidden_dim)
        #self.bond_encoder = BondEncoder(hidden_dim)
        # replace with a linear layer to project features to hidden_dim
        self.feature_projector = nn.Linear(input_dim, hidden_dim, bias=True)
        self.edge_feature_projector = nn.Linear(input_dim, hidden_dim, bias=True)
        ###########################################################################

        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList([ GatedGCNLSPELayer(hidden_dim, hidden_dim, dropout, self.batch_norm, self.residual) 
                    for _ in range(n_layers-1) ]) 
            self.layers.append(GatedGCNLSPELayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        else: 
            # NoPE or LapPE
            self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout, self.batch_norm, self.residual) 
                    for _ in range(n_layers-1) ]) 
            self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(out_dim+self.pos_enc_dim, out_dim)
        
        self.g = None              # For util; To be accessed in loss() function
        
    def forward(self, g, h, p, e, snorm_n):

        #h = self.atom_encoder(h)
        h = self.feature_projector(h)
        #e = self.bond_encoder(e)
        e = self.edge_feature_projector(e)

        if self.pe_init in ['rand_walk', 'lap_pe']:
            p = self.embedding_p(p)
            
        if self.pe_init == 'lap_pe':
            h = h + p
            p = None
        
        for conv in self.layers:
            h, p, e = conv(g, h, p, e, snorm_n)
            
        g.ndata['h'] = h
        
        if self.pe_init == 'rand_walk':
            p = self.p_out(p)
            g.ndata['p'] = p
            
        if self.use_lapeig_loss:
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p']**2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms+1e-6)            
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p
        
        if self.pe_init == 'rand_walk':
            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'],g.ndata['p']),dim=-1))
            g.ndata['h'] = hp

        ########################## Graph Classification Readout #######################################
        # if self.readout == "sum":
        #     hg = dgl.sum_nodes(g, 'h')
        # elif self.readout == "max":
        #     hg = dgl.max_nodes(g, 'h')
        # elif self.readout == "mean":
        #     hg = dgl.mean_nodes(g, 'h')
        # else:
        #     hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        #################################################################################################
            
        self.g = g # For util; To be accessed in loss() function
        
        if self.n_classes == 128:
            return_g = None # not passing PCBA graphs due to memory
        else:
            return_g = g
        ########################### Graph classification  ############################################
        #return self.MLP_layer(hg), return_g
        #################################################################################################

        ########################### Node Classification  ###############################################
        m = torch.nn.Sigmoid()
        return m(self.MLP_layer(h)), return_g
        #################################################################################################
        
    def loss(self, pred, labels):
        
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = torch.nn.BCELoss()(pred, labels)
        
        if self.use_lapeig_loss:
            raise NotImplementedError
        else:
            loss = loss_a
            
        return loss




