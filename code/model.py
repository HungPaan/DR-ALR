# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy.sparse as sps
from copy import deepcopy
# torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)



class mf(nn.Module):    
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        print('mf initialized')

    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])


        out = torch.sigmoid((user_emb * item_emb).sum(dim=1))

        return out  
    
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred 
    





class logistic_regression(nn.Module):    
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias=True)

        print('logistic_regression initialized')

    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])

        # concat
        z_emb = torch.cat([user_emb, item_emb], axis=1)

        out = torch.sigmoid(self.linear_1(z_emb))

        return torch.squeeze(out)        
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred 
    


        
class dr_alr(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard, non_linear='relu', final_bias=1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard

        
        self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
        self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   

        

        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)

        print('dr_alr initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop:(idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()

            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, gamma, G, pred_lr, impu_lr, prop_lr, pred_lamb, impu_lamb, prop_lamb, stop, tol)          
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):                
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # update model_pred
                ## propensity score
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size

                ## direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / inv_prop.detach().sum()
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                # update model_impu
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                pred_u = self.model_pred.predict(x_sampled) 
                imputation_y1 = self.model_impu(x_sampled)   
                
                imputation_loss_u = F.binary_cross_entropy(imputation_y1, pred_u.detach(), reduction='mean')
            
                xent_loss = F.binary_cross_entropy(pred.detach(), sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='sum')
                ips_loss = (xent_loss - imputation_loss)
                direct_loss = F.binary_cross_entropy(imputation_y1, pred_u.detach(), reduction='sum')
                dr_loss = (ips_loss + direct_loss) / inv_prop.detach().sum()
                
                imp_loss = torch.sum(((e_loss - e_hat_loss.detach() + imputation_loss_u - dr_loss.detach()) ** 2) * (inv_prop.detach() ** 2)) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                            
                epoch_loss += xent_loss.detach()
            

                
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                

        return epoch
            


             
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach()    




