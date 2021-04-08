# coding: utf-8
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import argparse
import numpy as np
from json import encoder
from model import AttnRnnModel
from train import input_data, run_model_train, run_model_test
import pickle
import time


def run(parameters):
    if parameters.model_mode == 'attn_RNN':
        model = AttnRnnModel(parameters=parameters).cuda()  if parameters.use_cuda else AttnRnnModel(parameters=parameters)
    criterion = nn.NLLLoss().cuda() if parameters.use_cuda else nn.NLLLoss()


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
                           weight_decay=parameters.L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step, 
                                                     factor=parameters.lr_decay, threshold=1e-3)

    data_train, train_idx = input_data(parameters.data_user, 'train')
    data_test, test_idx = input_data(parameters.data_user, 'test')


    metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {}}
    print(parameters.save_path.split('/'))
    result_path = parameters.save_path.split('/')[0]
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    tmp_path = 'checkpoint/'
    if not os.path.exists(result_path+'/'+tmp_path):
        os.mkdir(result_path+'/'+tmp_path)
    print(result_path+'/'+tmp_path)

    save_name_tmp_list = []

    for epoch in range(parameters.epoch_max): 
        st = time.time()
        data_train, train_idx = input_data(parameters.data_user, 'train')
        data_test, test_idx = input_data(parameters.data_user, 'test')



        model, avg_loss = run_model_train(data_train, train_idx, 'train', parameters.lr, parameters.clip, model, optimizer,
                                         criterion, parameters.model_mode, parameters.use_cuda, data_test, test_idx, parameters.list_history,parameters.list_history_traces)  
        
        print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, parameters.lr))
        metrics['train_loss'].append(avg_loss) 
        avg_loss, acc_all  = run_model_test(data_test, test_idx, 'test', parameters.lr, parameters.clip, model, optimizer, criterion, parameters.model_mode, parameters.use_cuda, data_train, train_idx, parameters.list_history,parameters.list_history_traces)
        print('Loss:{:.4f}'.format(avg_loss))
        print(acc_all)
        print('==>Test Acc (Check_ins):{:.4f},{:.4f},{:.4f}'.format(acc_all[0], acc_all[1], acc_all[2] ))
        print('==>Test Acc (Final_state):{:.4f},{:.4f},{:.4f} '.format(acc_all[3], acc_all[4], acc_all[5] ))
        print('==>Macro_P={:.4f},Macro_R={:.4f},Macro_F1={:.4f} '.format(acc_all[9], acc_all[10], acc_all[11] ))

        metrics['valid_loss'].append(avg_loss) 
        metrics['accuracy'].append(acc_all[5])
        if parameters.save_model:
            save_name_tmp = str(st)+'ep_' + str(epoch) + '.m' 
            save_name_tmp_list.append(save_name_tmp) 
            torch.save(model.state_dict(), args.save_path + tmp_path + save_name_tmp)

        scheduler.step(acc_all[5])

        lr_last = parameters.lr 
        parameters.lr = optimizer.param_groups[0]['lr'] 
        if parameters.save_model:
            if lr_last > parameters.lr:
                load_epoch = save_name_tmp_list[np.argmax(metrics['accuracy'])] 
                load_name_tmp = str(load_epoch)
                print(epoch, load_epoch)
                model.load_state_dict(torch.load(args.save_path + tmp_path + load_name_tmp))  
                print('load epoch={} model state'.format(load_epoch))
        
        if epoch >= 0:
           print('single epoch time cost:{}'.format(time.time() - st)) 


    mid = np.argmax(metrics['accuracy'])
    acc_all = metrics['accuracy'][mid]
    load_name_tmp =save_name_tmp_list[mid]
    model.load_state_dict(torch.load(args.save_path + tmp_path + load_name_tmp)) 
    argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.lr,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max,  'model_mode': args.model_mode}
    save_name = 'res'
    json.dump({'args': argv, 'metrics': metrics}, fp=open(args.save_path + save_name + '.rs', 'w'), indent=4)
    metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy': []} 
    for key in metrics_view:
        metrics_view[key] = metrics[key]
    json.dump({'args': argv, 'metrics': metrics_view}, fp=open(args.save_path + save_name + '.txt', 'w'), indent=4)
    torch.save(model.state_dict(), args.save_path + save_name + '.m')   
    return acc_all


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)            
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--loc_emb_size', type=int, default=64)
    parser.add_argument('--uid_emb_size', type=int, default=32)
    parser.add_argument('--tim_emb_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)#128
    parser.add_argument('--dropout_p', type=float, default=0.6)#
    parser.add_argument('--data_name', type=str, default='gowalla')
    parser.add_argument('--lr', type=float, default=0.005)#0.005
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.5)#
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=20 * 1e-5, help=" weight decay (L2 penalty)")#20 * 1e-5
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=40)
    parser.add_argument('--rnn_type', type=str, default='BILSTM', choices=['LSTM', 'GRU', 'RNN','BILSTM'])
    parser.add_argument('--attn_type', type=str, default='general', choices=['general', 'concat', 'dot'])
    parser.add_argument('--strategies_type', type=str, default='AVE-sdot', choices=['AVE-sdot', 'AVE-dot', 'MAX-dot','MAX-sdot'])
    parser.add_argument('--save_path', type=str, default='results/')
    parser.add_argument('--model_mode', type=str, default='attn_RNN',choices=['attn_RNN'])
    parser.add_argument('--GPU_number',type=str, default='6')
    parser.add_argument('--file_name',type=str, default='final_108_1month_Tokyo')
    args = parser.parse_args()
    parameters = args

    os.environ["CUDA_VISIBLE_DEVICES"] = parameters.GPU_number  
    print("start!")
    print('GPU_number',os.environ["CUDA_VISIBLE_DEVICES"])
    print(args)
    with open('data/gowalla_processed'+parameters.file_name+'.pk', 'rb') as f:
         parameters.data_user = pickle.load(f)
    print('Have read the data_user!')

    parameters.list_history = pickle.load(open('data/list_history'+parameters.file_name+'.pk', 'rb'))
    print('Have read the list_history!')
    parameters.list_history_traces = pickle.load(open('data/list_history_traces'+parameters.file_name+'.pk', 'rb'))

    parameters.uid_size = len(parameters.data_user) + 1
    print('model_mode=',parameters.model_mode)
    parameters.loc_size = parameters.data_user[1]['pid_number'] + 1
    parameters.tim_size = 2 * 24 + 5
    parameters.use_cuda = True
    print("use_cuda=",parameters.use_cuda)
    print('loc_size=',parameters.loc_size, 'uid_size=',parameters.uid_size, 'tim_size=', parameters.tim_size)
    print('trace_size=',len(parameters.list_history))
    final_acc = run(parameters)
    print('final_acc:{:.4f}'.format(final_acc))
    print('Done!!!')
    