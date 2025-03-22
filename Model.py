from torch import nn
import torch.nn.functional as F
import torch
from Params import args
from copy import deepcopy
import numpy as np
import math
import scipy.sparse as sp
from Utils.Utils import contrastLoss, calcRegLoss, pairPredict
import time
import torch_sparse
import pickle

init = nn.init.xavier_uniform_


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.uEmbeds = torch.nn.Embedding(
            num_embeddings=args.user, embedding_dim=args.latdim)
        self.iEmbeds = torch.nn.Embedding(
            num_embeddings=args.item, embedding_dim=args.latdim)
        nn.init.normal_(self.uEmbeds.weight, std=0.1)
        nn.init.normal_(self.iEmbeds.weight, std=0.1)
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
        self.gcn = GCNLayer()
        self.gcn_social = GCNLayer_social()
        self.MLP0 = MLP_0(args.latdim * 2, 2)
        self.MLP1 = MLP_1(args.latdim * 2, 2)
        self.MLP2 = MLP_2(args.latdim * 2, 2)
        self.MLP3 = MLP_3(args.latdim * 2, 2)
        self.MLP4 = MLP_4(args.latdim * 2, 2)
        self.MLP01 = MLP_01(args.latdim * 2, 2)
        self.MLP11 = MLP_11(args.latdim * 2, 2)
        self.MLP21 = MLP_21(args.latdim * 2, 2)
        self.MLP31 = MLP_31(args.latdim * 2, 2)
        self.MLP41 = MLP_41(args.latdim * 2, 2)
        self.MLPL1 = MLP_L1(args.latdim * 5, 5)
        self.MLPL2 = MLP_L2(args.latdim * 5, 5)

    def forward_gcn(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds.weight, self.iEmbeds.weight], dim=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)
        return mainEmbeds[:args.user], mainEmbeds[args.user:]

    def forward_graphcl(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds.weight, self.iEmbeds.weight], dim=0)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)
        return mainEmbeds

    def forward_fusion2(self, interaction_adj, socail_adj, a, b):  
        iniEmbeds = torch.concat((self.uEmbeds.weight, self.iEmbeds.weight), dim=0)
        iniEmbeds_s = iniEmbeds[:args.user]
        embedsLst_interaction = [iniEmbeds]
        abation = [iniEmbeds]
        embedsLst_social = [iniEmbeds_s]
        embedsLst = []
        if a == 1:
            for i in range(args.gnn_layer):
                embeds_social = self.gcn(socail_adj, embedsLst_social[-1])
                embedsLst_social.append(embeds_social)
            for i in range(args.gnn_layer):
                embeds_inter = self.gcn(interaction_adj, abation[-1])
                abation.append(embeds_inter)

            uu_0 = embedsLst_social[0]
            ui_0 = embedsLst_interaction[0]
            inter_u = ui_0[:args.user]
            m0 = self.MLP0(uu_0, inter_u)
            m01 = torch.reshape(m0[:, 0], [args.user, 1])
            m02 = torch.reshape(m0[:, 1], [args.user, 1])
            embeds = torch.cat((uu_0 * m01 + inter_u * m02, ui_0[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction[0] = embeds
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)

            uu_1 = embedsLst_social[1]
            ui_1 = embedsLst_interaction[1]
            inter_u = ui_1[:args.user]
            m1 = self.MLP1(uu_1, inter_u)
            m11 = torch.reshape(m1[:, 0], [args.user, 1])
            m12 = torch.reshape(m1[:, 1], [args.user, 1])
            embeds = torch.cat((uu_1 * m11 + inter_u * m12, ui_1[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)

            uu_2 = embedsLst_social[2]
            ui_2 = embedsLst_interaction[2]
            inter_u = ui_2[:args.user]
            m2 = self.MLP2(uu_2, inter_u)
            m21 = torch.reshape(m2[:, 0], [args.user, 1])
            m22 = torch.reshape(m2[:, 1], [args.user, 1])
            embeds = torch.cat((uu_2 * m21 + inter_u * m22, ui_2[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)
            #
            uu_3 = embedsLst_social[3]
            ui_3 = embedsLst_interaction[3]
            inter_u = ui_3[:args.user]
            m3 = self.MLP3(uu_3, inter_u)
            m31 = torch.reshape(m3[:, 0], [args.user, 1])
            m32 = torch.reshape(m3[:, 1], [args.user, 1])
            embeds = torch.cat((uu_3 * m31 + inter_u * m32, ui_3[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)



        else:
            for i in range(args.gnn_layer):
                embeds_social = self.gcn(socail_adj, embedsLst_social[-1])
                embedsLst_social.append(embeds_social)
            for i in range(args.gnn_layer):
                embeds_inter = self.gcn(interaction_adj, abation[-1])
                abation.append(embeds_inter)

            uu_0 = embedsLst_social[0]
            ui_0 = embedsLst_interaction[0]
            inter_u = ui_0[:args.user]
            m0 = self.MLP01(uu_0, inter_u)
            m01 = torch.reshape(m0[:, 0], [args.user, 1])
            m02 = torch.reshape(m0[:, 1], [args.user, 1])
            embeds = torch.cat((uu_0 * m01 + inter_u * m02, ui_0[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction[0] = embeds
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)

            uu_1 = embedsLst_social[1]
            ui_1 = embedsLst_interaction[1]
            inter_u = ui_1[:args.user]
            m1 = self.MLP11(uu_1, inter_u)
            m11 = torch.reshape(m1[:, 0], [args.user, 1])
            m12 = torch.reshape(m1[:, 1], [args.user, 1])
            embeds = torch.cat((uu_1 * m11 + inter_u * m12, ui_1[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)

            uu_2 = embedsLst_social[2]
            ui_2 = embedsLst_interaction[2]
            inter_u = ui_2[:args.user]
            m2 = self.MLP21(uu_2, inter_u)
            m21 = torch.reshape(m2[:, 0], [args.user, 1])
            m22 = torch.reshape(m2[:, 1], [args.user, 1])
            embeds = torch.cat((uu_2 * m21 + inter_u * m22, ui_2[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)
            #
            uu_3 = embedsLst_social[3]
            ui_3 = embedsLst_interaction[3]
            inter_u = ui_3[:args.user]
            m3 = self.MLP31(uu_3, inter_u)
            m31 = torch.reshape(m3[:, 0], [args.user, 1])
            m32 = torch.reshape(m3[:, 1], [args.user, 1])
            embeds = torch.cat((uu_3 * m31 + inter_u * m32, ui_3[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)

        main_embeds = sum(embedsLst_interaction)
        out = sum(abation)
        return main_embeds, out

    def loss_graphcl(self, x1, x2, users, items):  
        user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)
        user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)

        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
                                                                                    all_embs2_abs)
        sim_matrix = torch.exp(sim_matrix / T)  
        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]  
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  
        loss = - torch.log(loss)  


        return loss

    def loss_graphcl_2(self, x1, x2, users, items):  # 对比损失函数
        T = args.temp
        user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)
        user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)

        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
                                                                                    all_embs2_abs)

        sim_matrix = torch.exp(sim_matrix / T)  
        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]  

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  
        loss = - torch.log(loss)  
        loss_user = loss[:len(users)]
        loss_item = loss[len(items)]
        return loss_user, loss_item

class MLP_L1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_L1, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, concatenated):
        
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_L2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_L2, self).__init__()
       
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, concatenated):
        
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_0(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_0, self).__init__()
       
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_1, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
       
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_2, self).__init__()
       
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
       
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_3, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
       
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_4, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output



class MLP_01(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_01, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_11(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_11, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
    
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_21(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_21, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_31(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_31, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_41(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_41, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output


class EmbeddingMapper(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers):
        super(EmbeddingMapper, self).__init__()
        self.num_layers = num_layers
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)

        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        
        self.fc_last = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, x):
        x = F.elu(self.fc1(x))

        for layer in self.hidden_layers:
            x = F.elu(layer(x))

        x_mapped = self.fc_last(x)

        return x_mapped



class GCNLayer(nn.Module):  
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):

        return torch.spmm(adj, embeds)

class GCNLayer_social(nn.Module):
    def __init__(self):
        super(GCNLayer_social, self).__init__()

    def forward(self, adj, embeds):

        return torch.spmm(adj, embeds)



class vgae_encoder(Model):
    def __init__(self):
        super(vgae_encoder, self).__init__()
        hidden = args.latdim
        self.encoder_mean = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.encoder_std = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                         nn.Softplus())

    def forward(self, adj):  
        x = self.forward_graphcl(adj)
        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).cuda()
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std


class vgae_decoder(nn.Module):
    def __init__(self, hidden=args.latdim):
        super(vgae_decoder, self).__init__()
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                     nn.Linear(hidden, 1))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
        x_user, x_item = torch.split(x, [args.user, args.item], dim=0)

        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda())
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda())
        loss_rec = loss_edge_pos + loss_edge_neg

        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean ** 2 - x_std ** 2).sum(dim=1)

        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
        regLoss = calcRegLoss(encoder) * args.reg

        beta = 0.1
        loss = (loss_rec + beta * kl_divergence.mean()).mean()
        return loss


class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, users, items, neg_items):
        x, x_mean, x_std = self.encoder(data)
        loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)
        return loss

    def generate(self, data, edge_index, adj):
        x, _, _ = self.encoder(data)

        edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))

        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        edge_pred = edge_pred[:, 0]
        mask = ((edge_pred + 0.5).floor()).type(torch.bool)

        newVals = vals[mask]

        newVals = newVals / (newVals.shape[0] / edgeNum[0])
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
