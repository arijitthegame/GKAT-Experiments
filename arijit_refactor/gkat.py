# pylint: skip-file

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GKATLayer(nn.Module):
    def __init__(self, 
        in_dim, 
        out_dim, 
        feat_drop=0.0, 
        attn_drop=0.0, 
        alpha=0.2, 
        agg_activation=F.elu
        ):
        super(GKATLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.alpha = alpha
        self.agg_activation = agg_activation

        self.feat_dropout = nn.Dropout(self.feat_drop)  
        self.attn_dropout = nn.Dropout(self.attn_drop)  
        
        self.fc_Q = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.fc_K = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.fc_V = nn.Linear(self.in_dim, self.out_dim, bias=False)
        
        self.softmax = nn.Softmax(dim = 1) #where is this used?
        
    def forward(self, feat, bg, counting_attn): #where is bg used in this forward
        h = self.feat_dropout(feat)

        Q = self.fc_Q(h).reshape((h.shape[0], -1))
        K = self.fc_K(h).reshape((h.shape[0], -1))
        V = self.fc_V(h).reshape((h.shape[0], -1))
        
        logits = torch.matmul( Q, torch.transpose(K,0,1) ) / math.sqrt(Q.shape[1]) #no softmax
        logits = self.attn_dropout(logits)

        maxes = torch.max(logits, 1, keepdim=True)[0]
        logits =  logits - maxes
        
        a_nomi = torch.matmul(torch.exp(logits), counting_attn)
        a_deno = torch.sum(a_nomi, 1, keepdim=True)
        a_nor = a_nomi/(a_deno+1e-9)

        ret = torch.mm(a_nor, V)
        if self.agg_activation is not None:
            ret = self.agg_activation(ret)

        return ret


class GKATClassifier(nn.Module):
    def __init__(self, 
                in_dim, 
                hidden_dim, 
                num_heads, 
                n_classes, 
                feat_drop_= 0.0,
                attn_drop_=0.0,
                agg_activation = F.elu,
                normalize = True
                ):
        super(GKATClassifier, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim #list of 2 integers
        self.n_classes = n_classes
        self.feat_drop = feat_drop_
        self.attn_drop = attn_drop_
        self.in_dim = in_dim
        self.agg_activation = agg_activation
        self.normalize = normalize


        self.layers = nn.ModuleList([
            nn.ModuleList([GKATLayer(self.in_dim, self.hidden_dim[0], feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation) 
                                    for _ in range(num_heads)]),
            nn.ModuleList([GKATLayer(hidden_dim[0] * num_heads, hidden_dim[-1], feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation)) 
                        for _ in range(1)])
        ])

        self.classify = nn.Linear(hidden_dim[-1] * 1, n_classes) #why this times 1
        self.softmax = nn.Softmax(dim = 1) #where is this used

    def forward(self, feats, counting_attn):
        
        if self.normalize:
            mean_ = torch.mean(feats, dim=1, keepdim=True)
            std_ = torch.std(feats, dim=1, keepdim=True)
            feats = (feats - mean_)/(std_+1e-9)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(feats, counting_attn))   
            feats = torch.squeeze(torch.cat(all_h, dim=1))

        return self.classify(feats)