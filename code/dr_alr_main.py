from tensorboardX import SummaryWriter
import time
from dataset import load_data, load_data_ori
from utils import set_seed, check_dir, generate_total_sample, ndcg_func, recall_func, precision_func
import torch
import numpy as np
import csv
from parse import parse_args
from model import dr_alr
import pandas as pd
from sklearn.metrics import roc_auc_score
import optuna
import os
import itertools
mse_func = lambda x,y: np.mean((x-y)**2)



if __name__ == '__main__':
    args = parse_args()
    print("args", args)
    if args.debias_name == 'dr_alr':
        print(f'{args.debias_name}')
    else:
        print('wrong debias method')
        exit()

    set_seed(args.seed)
    args.batch_size, args.batch_size_prop, args.gamma, args.G, args.pred_lr, args.pred_lamb, args.impu_lr, args.impu_lamb, args.prop_lr, args.prop_lamb = map(float, f'{args.hyper_str}'.split('_'))  

    args.embedding_k = int(args.embedding_k)
    args.batch_size = int(args.batch_size)
    args.batch_size_prop = int(args.batch_size_prop)
    args.G = int(args.G)
    
    file_name = f"{args.data_name}{args.thres}{args.train_rate}{args.val_rate}_{args.debias_name}_{args.pred_model_name}_{args.prop_model_name}_ce_{args.load_param_type}_impu_copy{args.copy_model_pred}_{args.embedding_k}_grad_type{args.grad_type}_{args.num_epochs}_{args.batch_size_prop}_{args.batch_size}_{args.sever}_{args.device}_{args.seed}_optuna_tune_on_{args.tune_type}_{args.auc_type}_auc_ex{args.ex_idx}"
    args.file_name = file_name
    print("file_name", file_name)

    aug_load_param_type = file_name

    if args.train_rate != 1.0:
        num_users, num_items, x_train, x_val, x_test, y_train, y_val, y_test = load_data(args.data_name, args.data_path, args.thres, args.train_rate, args.val_rate)
    else:
        num_users, num_items, x_train, x_val, x_test, y_train, y_val, y_test = load_data_ori(args.data_name, args.data_path, args.thres, args.val_rate)
    print('positive rate', y_train.sum() / float(len(y_train)), y_test.sum() / float(len(y_test)))

    x_train_tensor = torch.from_numpy(x_train).long().to(args.device)
    y_train_tensor = torch.from_numpy(y_train).float().to(args.device)
    x_val_tensor = torch.from_numpy(x_val).long().to(args.device)
    y_val_tensor = torch.from_numpy(y_val).float().to(args.device)
    x_test_tensor = torch.from_numpy(x_test).long().to(args.device)
    y_test_tensor = torch.from_numpy(y_test).float().to(args.device)
    
    
    x_all = generate_total_sample(num_users, num_items).to(args.device)

    obs = torch.sparse.FloatTensor(torch.cat([x_train_tensor[:, 0].unsqueeze(dim=0), x_train_tensor[:, 1].unsqueeze(dim=0)], dim=0), torch.ones_like(y_train_tensor), torch.Size([num_users, num_items])).to_dense().reshape(-1)
    
    
        
    # check_dir(f'../metric/{args.debias_name}/', '_')
    # check_dir(f'../model_param/{args.debias_name}/', '_')
    # check_dir(f'../optuna_storage/{args.debias_name}/', '_')
    print("data info")
    print("num_users", num_users)
    print("num_items", num_items)
    print('x_train_tensor', x_train_tensor, x_train_tensor.shape)
    print('y_train_tensor', y_train_tensor, y_train_tensor.shape)
    print('x_val_tensor', x_val_tensor, x_val_tensor.shape)
    print('y_val_tensor', y_val_tensor, y_val_tensor.shape)
    print('x_test_tensor', x_test_tensor, x_test_tensor.shape)
    print('y_test_tensor', y_test_tensor, y_test_tensor.shape)
    print('x_all', x_all, x_all.shape)
    print('obs', obs, obs.shape)
    print('check_x_all', obs[x_train_tensor[:, 0]*num_items+x_train_tensor[:, 1]], obs[x_train_tensor[:, 0]*num_items+x_train_tensor[:, 1]].sum())
    

    
    hyper_param = str(args.batch_size) + '_' + str(args.batch_size_prop) + '_' + str(args.gamma) + '_' + str(args.G) + '_' + str(args.pred_lr) + '_' + str(args.pred_lamb) + '_' + str(args.impu_lr) + '_' + str(args.impu_lamb)  + '_' + str(args.prop_lr) + '_' + str(args.prop_lamb)

    print("hyper_param", hyper_param)

    if args.is_tensorboard:
        tb_log = SummaryWriter(f"{args.tensorborad_path}{file_name}/{hyper_param}")

    else:
        tb_log = None
    
    set_seed(args.seed)
    res_list = []
    final_res_list = []

    for se in range(2024, 2024+5):
        set_seed(se)
        print('seed', se)
        res_list = []
        
        my_model = dr_alr(num_users, num_items, args.pred_model_name, args.prop_model_name, aug_load_param_type, args.copy_model_pred, args.embedding_k, args.batch_size_prop, args.batch_size, args.device, args.is_tensorboard)
        
        log_prop_epoch = my_model._compute_IPS(tb_log, x_all, obs, num_epochs=200, prop_lr=args.prop_lr, prop_lamb=args.prop_lamb)
        

        log_epoch = my_model.fit(tb_log, x_all, obs, x_train_tensor, y_train_tensor, grad_type=args.grad_type, num_epochs=args.num_epochs, gamma=args.gamma, G=args.G, pred_lr=args.pred_lr, impu_lr=args.impu_lr, prop_lr=args.prop_lr2, pred_lamb=args.pred_lamb, impu_lamb=args.impu_lamb, prop_lamb=args.prop_lamb2)

        
        val_pred = my_model.predict(x_val_tensor).detach().cpu().numpy()
        log_val_auc = roc_auc_score(y_val, val_pred)
        
        test_pred = my_model.predict(x_test_tensor).detach().cpu().numpy()
        log_test_auc = roc_auc_score(y_test, test_pred)
        
        if 'kuai' not in args.data_name:
            log_test_ndcg = ndcg_func(my_model, x_test, y_test, device=args.device)
            log_test_recall = recall_func(my_model, x_test, y_test, device=args.device)
            log_test_precision = precision_func(my_model, x_test, y_test, device=args.device)
            
            res_list = [file_name, hyper_param, log_epoch, log_val_auc, log_test_auc, np.mean(log_test_ndcg['ndcg_5']), np.mean(log_test_recall['recall_5'])]
            column_names = ["file_name", "hyper_param", "log_epoch", "val_auc", "auc", "ndcg_5", "recall_5"]

        else:
            top_k_list = [50]
            log_test_ndcg = ndcg_func(my_model, x_test, y_test, device=args.device, top_k_list=top_k_list)
            log_test_recall = recall_func(my_model, x_test, y_test, device=args.device, top_k_list=top_k_list)
            log_test_precision = precision_func(my_model, x_test, y_test, device=args.device, top_k_list=top_k_list)
            
            res_list = [file_name, hyper_param, log_epoch, log_val_auc, log_test_auc, np.mean(log_test_ndcg['ndcg_50']), np.mean(log_test_recall['recall_50']), np.mean(log_test_precision['precision_50'])]
            
            column_names = ["file_name", "hyper_param", "log_epoch", "val_auc", "auc", 'ndcg_50', 'recall_50', 'precision_50']
            
        print('res_list', res_list)
        final_res_list.append(res_list[3:])

        

        if args.is_tensorboard:
            tb_log.close()
    

    final_res_list = np.array(final_res_list)
    mean_res = np.mean(final_res_list, axis=0)
    std_res = np.std(final_res_list, axis=0)
    print('MEAN validation auc, test auc, test ndcg_5, test recall_5', mean_res)
    print('STD validation auc, test auc, test ndcg_5, test recall_5', std_res)