# pylint: skip-file

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

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
        self.alpha = alpha #where is this used
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
                attn_drop_= 0.0,
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
            nn.ModuleList([
                    GKATLayer(self.in_dim, self.hidden_dim[0], \
                        feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation) 
                                for _ in range(self.num_heads)]
                                    ),
            nn.ModuleList([
                    GKATLayer(self.hidden_dim[0] * self.num_heads, self.hidden_dim[-1], \
                        feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation) 
                            for _ in range(1)]
                        )
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


class GKATMultiHead(nn.Module):
    def __init__(self, 
                in_dim, 
                out_dim,
                num_heads,
                feat_drop= 0.0,
                attn_drop= 0.0,
                agg_activation = F.elu,
                ):
        super(GKATMultiHead, self).__init__()

        self.num_heads = num_heads
        self.out_dim = out_dim
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.in_dim = in_dim
        self.agg_activation = agg_activation

        self.feat_dropout = nn.Dropout(self.feat_drop)
        self.attn_dropout = nn.Dropout(self.attn_drop)

        assert(self.out_dim % self.num_heads == 0)
        self.dim_head = self.out_dim/self.num_heads

        self.to_qvk = nn.Linear(self.in_dim, self.out_dim * 3, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, feat, counting_attn, bg=None): #where is bg used in this forward
        x = self.feat_dropout(feat)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # Project to Q, K, V
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # Step 2
        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be:
        # [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d ', k=3, h=self.num_heads))

        # Step 3
        # resulted shape will be: [batch, heads, tokens, tokens]
        logits = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor

        logits = self.attn_dropout(logits)

        logits =  logits - torch.max(logits, 3, keepdim=True)[0] 

        if counting_attn.dim() == 2:
            counting_attn = repeat(counting_attn, 'l w -> h l w', h=self.num_heads) #repeat along heads
            counting_attn = counting_attn.unsqueeze(0) #add batch dim

        elif counting_attn.dim() == 3:
            counting_attn = repeat(counting_attn, 'b l w -> b h l w', h=self.num_heads) #repeat along heads 
        #TODO: check if this is correct
        else:
            raise ValueError("counting_attn should be 2 or 3 dim")

        # Step 4 Calculate the masked logits
        logit_mask = torch.einsum('b h i j, b h j k -> b h i k', torch.exp(logits), counting_attn)
        logit_mask  = logit_mask/(torch.sum(logit_mask, 3, keepdim=True) + 1e-9)


        # Step 4. Calc result per batch and per head h
        out = torch.einsum('b h i j , b h j d -> b h i d', logit_mask, v)

        # Step 5. Re-compose: merge heads with dim_head d
        out = rearrange(out, "b h t d -> b t (h d)")

        if feat.dim() == 2:
            out = out.squeeze()

        if self.agg_activation is not None:
            out = self.agg_activation(out)

        return out


class GKATMultiHeadClassifier(nn.Module):
    def __init__(self, 
                in_dim, 
                out_dim, 
                num_heads, 
                num_layers,
                n_classes, 
                feat_drop= 0.0,
                attn_drop= 0.0,
                agg_activation = F.elu,
                normalize = True
                ):
        super(GKATMultiHeadClassifier, self).__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.out_dim = out_dim #list of m integers, output dims of the layers, where m is the number of GKAT layers
        self.n_classes = n_classes
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.in_dim = in_dim
        self.agg_activation = agg_activation
        self.normalize = normalize

        self.layers = nn.ModuleList([
            GKATMultiHead(self.in_dim, self.hidden_dim[0], self.num_heads,\
                        feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation) 
                                for _ in range(self.num_layers-1)]
                                    )
        self.final_gkat_layer = GKATMultiHead(self.hidden_dim[0] , self.hidden_dim[-1], num_heads=1, \
                        feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation)
            

        self.classify = nn.Linear(hidden_dim[-1] * 1, n_classes) #why this times 1
        self.softmax = nn.Softmax(dim = 1) #where is this used

    def forward(self, feats, counting_attn, bg=None):
        
        if self.normalize:
            if feats.dim() == 2:
                mean_ = torch.mean(feats, dim=1, keepdim=True)
                std_ = torch.std(feats, dim=1, keepdim=True)
                feats = (feats - mean_)/(std_+1e-9)

            elif feats.dim() == 3:
                mean_ = torch.mean(feats, dim=2, keepdim=True)
                std_ = torch.std(feats, dim=2, keepdim=True)
                feats = (feats - mean_)/(std_+1e-9)

            else:
                raise ValueError("feats should be 2 or 3 dim")

        for i, gnn in enumerate(self.layers):
            feats = gnn(feats, counting_attn, bg=None)

        feats = self.final_gkat_layer(feats,counting_attn, bg=None)

            
        return self.classify(feats.squeeze())
