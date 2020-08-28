# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable
import numpy as np
import cPickle as pickle
from collections import deque, Counter
from masked_cross_entropy import masked_cross_entropy
import time

def input_data(data_user, mode):
    data_train = {}
    train_idx = {}
    for u in data_user.keys():
        sessions = data_user[u]['sessions_' + mode]
        train_id = data_user[u]['id_' + mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            session = sessions[i]
            trace = {}


            loc_np = [s[0] for s in session]
            tim_np = [s[1] for s in session]
            target = [u]*len(session)

            trace['loc'] = loc_np
            trace['tim'] = tim_np
            trace['target'] = target
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def get_acc(scores_all, target_all, target_lengths, TEST_DIC):
    acc_all = []
    batch_size = target_all.size(0)

    for m in range(batch_size):
        acc = [0] * 9
        target = target_all[m].data.cpu().numpy()
        val, idxx = scores_all[m].data.topk(10, 1)
        predx = idxx.cpu().numpy()
        target_length = target_lengths[m]

        TEST_DIC[target[0]][2] += 1
        
        for i in range(target_length):
            t = target[i]  
            p = predx[i]
            if t in p[:10] and t > 0 and i < target_length:
                acc[0] += 1
            if t in p[:5] and t > 0 and i < target_length:
                acc[1] += 1
            if t == p[0] and t > 0 and i < target_length:
                acc[2] += 1

        acc[0] = float(acc[0])/target_length
        acc[1] = float(acc[1])/target_length
        acc[2] = float(acc[2])/target_length

        if predx[target_length-1][0] in TEST_DIC:
            TEST_DIC[predx[target_length-1][0]][1] += 1 
        if predx[target_length-1][0] not in TEST_DIC:
            print('user_number=',predx[target_length-1][0])

        if target[target_length-1] in predx[target_length-1][:10]:
            acc[3] = 1
        if target[target_length-1] in predx[target_length-1][:5]:
            acc[4] = 1
        if target[target_length-1] == predx[target_length-1][0]: 
            acc[5] = 1
            TEST_DIC[predx[target_length-1][0]][0] += 1 


        acc[3] = float(acc[3])
        acc[4] = float(acc[4])
        acc[5] = float(acc[5])
        acc_all.append(acc)

    return acc_all, TEST_DIC

def pad_seq(seq, max_length,loc_size):
    seq2 = seq + [loc_size for i in range(max_length - len(seq))]
    return seq2

def run_model_train(data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2, use_cuda, data_test, test_idx, list_all,history_traces):
    run_queue = list_all
    queue_len = len(list_all)
    total_loss = []


    history_loc = []
    history_tim = []
    history_uid = []
    history_count = []
    target_lengths = []
    target_var = []
    target1 = []
    scores = []
   
    flag = 0 
    batch_traj_loc = []
    batch_traj_tim = []
    history = []
    batch_size = 16
    run_batch_number = 0

    for c in range(queue_len):
        
        optimizer.zero_grad()
        u = run_queue[c][1]
        i = run_queue[c][2]
        target = []
        tim = []
        loc = []
        
        if run_queue[c][3] == 'train':
            loc = data[u][i]['loc']
            tim = data[u][i]['tim']
            target = data[u][i]['target']
        else:
            loc = data_test[u][i]['loc']
            tim = data_test[u][i]['tim']
            target = data_test[u][i]['target']

        if mode2 == 'attn_RNN':
            history = []
            if 24 < tim[0] < 48 : 
                time_mode = 'weekend'
            else:
                time_mode = 'weekday'

            if run_queue[c][3] == 'test':
                continue
            history = []


            if len(batch_traj_loc) < batch_size:
                t= len(batch_traj_loc)
                for trace in history_traces[c]:
                    if trace != []:
                        history.extend([(trace[0],trace[1],trace[2])])    
                if len(history)== 0 :continue
                history = sorted(history,key = lambda x:x[0],reverse = False)
                history_loc.append([s[0] for s in history])
                history_tim.append([s[1] for s in history])
                history_uid.append([s[2] for s in history])
                history_count.append([1])
                history_tl = [(s[0],s[1]) for s in history]
                last_tl = history_tl[0]
                for a in history_tl[1:]:
                    if cmp(a, last_tl) == 0:
                        history_count[t][-1] += 1
                    else:
                        history_count[t].append(1)

                        last_tl = a

                batch_traj_loc.append(loc)
                batch_traj_tim.append(tim)
                target1.append(target)
                current_time = run_queue[c][0]
                continue

            seq_pairs = zip(batch_traj_loc, batch_traj_tim, target1)
            batch_traj_loc, batch_traj_tim, target1 = zip(*seq_pairs)
            input_lengths = [len(s) for s in batch_traj_loc]
            input_padded_loc = [pad_seq(s, max(input_lengths), 0) for s in batch_traj_loc]
            input_padded_tim = [pad_seq(s, max(input_lengths), 0) for s in batch_traj_tim]
            input_var_loc = Variable(torch.LongTensor(input_padded_loc))
            input_var_tim = Variable(torch.LongTensor(input_padded_tim))
            
            target_lengths = [len(s) for s in target1]
            target_padded = [pad_seq(s, max(target_lengths), 0) for s in target1]
            target_var = Variable(torch.LongTensor(target_padded))#batch x seq
            loc_padded = [pad_seq(s, max(history_lengths), 0) for s in history_loc]    
            history_loc = Variable(torch.LongTensor(loc_padded))

            uid_padded = [pad_seq(s, max(history_lengths), 0) for s in history_uid] 
            history_uid = Variable(torch.LongTensor(uid_padded))

            tim_padded = [pad_seq(s, max(history_lengths), 0) for s in history_tim] 
            history_uid = Variable(torch.LongTensor(tim_padded))
            if use_cuda:
                input_var_loc = input_var_loc.cuda()
                input_var_tim = input_var_tim.cuda()
                target_var = target_var.cuda()
                history_loc = history_loc.cuda()
                history_uid = history_uid.cuda()
                history_tim = history_tim.cuda()
            model.train()
            scores = model(input_var_loc, input_var_tim, input_lengths, history_loc, history_tim, history_uid, history_count)
            batch_traj_tim = []
            batch_traj_loc = []
            target1 = []
            history = []
            history_loc = []
            history_tim = []
            history_uid = []
            history_count = []

        loss = masked_cross_entropy(
        scores.contiguous(), 
        target_var.contiguous(), 
        target_lengths)
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            for p in model.parameters():
                if p.requires_grad:
                    p.data.add_(-lr, p.grad.data)
        except:
            pass
        optimizer.step()
        total_loss.append(loss.data.cpu().numpy())

        if run_batch_number!=0 and run_batch_number%4000 == 0:
            print('Loss:{:.4f} '.format(np.mean(total_loss[run_batch_number-5:run_batch_number], dtype=np.float64)))


    avg_loss = np.mean(total_loss, dtype=np.float64)  
    return model, avg_loss


def run_model_test(data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2, use_cuda, data_train, train_idx, list_all,history_traces):
    total_loss = []
    queue_len = len(list_all)
    TEST_DIC = {}
    run_queue = list_all
    for line in run_queue:
        if line[1] not in TEST_DIC:
            TEST_DIC[line[1]]=[0,0,0]
    TEST_DIC[0]=[0,0,0]
    users_acc = []

    history = []
    
    history_loc = []
    history_tim = []
    history_uid = []
    history_count = []
    target_lengths = []
    target_var = []
    
    target1 = []
    target = []
    scores = []

    batch_traj_loc = []
    batch_traj_tim = []
    batch_size = 1
    run_batch_number = 0
    target_lengths = []
    input_lengths = []
    target_lengths = []
    target_var = []
    

    for c in range(queue_len):
        optimizer.zero_grad()
        u = run_queue[c][1]
        i = run_queue[c][2]

        if run_queue[c][3] == 'test':
            loc = data[u][i]['loc']
            tim = data[u][i]['tim']
            target = data[u][i]['target']
        else:
            loc = data_train[u][i]['loc']
            tim = data_train[u][i]['tim']
            target = data_train[u][i]['target']


            
        if mode2 == 'attn_RNN':
            if 24 < tim[0] < 48 : 
                time_mode = 'weekend'
            else:
                time_mode = 'weekday'
            
            if run_queue[c][3] == 'train':
                continue
            history = []
            if len(batch_traj_loc) < batch_size:
                t= len(batch_traj_loc)
                for trace in history_traces[c]:
                    if trace != []:
                        history.extend([(trace[0],trace[1],trace[2])])    
                if len(history)== 0 :continue
                history = sorted(history,key = lambda x:x[0],reverse = False)
                history_loc.append([s[0] for s in history])
                history_tim.append([s[1] for s in history])
                history_uid.append([s[2] for s in history])
                history_count.append([1])
                history_tl = [(s[0],s[1]) for s in history]
                last_tl = history_tl[0]
                for a in history_tl[1:]:
                    if cmp(a, last_tl) == 0:
                        history_count[t][-1] += 1
                    else:
                        history_count[t].append(1)

                        last_tl = a
                batch_traj_loc.append(loc)
                batch_traj_tim.append(tim)
                target1.append(target)
                current_time = run_queue[c][0]
                continue

            seq_pairs = zip(batch_traj_loc, batch_traj_tim, target1)
            batch_traj_loc, batch_traj_tim, target1 = zip(*seq_pairs)
            input_lengths = [len(s) for s in batch_traj_loc]
            input_loc = [pad_seq(s, max(input_lengths), 0) for s in batch_traj_loc]
            input_tim = [pad_seq(s, max(input_lengths), 0) for s in batch_traj_tim]
            input_var_loc = Variable(torch.LongTensor(input_loc))
            input_var_tim = Variable(torch.LongTensor(input_tim))
            
            target_lengths = [len(s) for s in target1]
            target_padded = [pad_seq(s, max(target_lengths), 0) for s in target1]
            target_var = Variable(torch.LongTensor(target_padded))
            history_lengths = [len(s) for s in history_loc]
            loc_padded = [pad_seq(s, max(history_lengths), 0) for s in history_loc]    
            history_loc = Variable(torch.LongTensor(loc_padded))

            uid_padded = [pad_seq(s, max(history_lengths), 0) for s in history_uid] 
            history_uid = Variable(torch.LongTensor(uid_padded))

            tim_padded = [pad_seq(s, max(history_lengths), 0) for s in history_tim] 
            history_uid = Variable(torch.LongTensor(tim_padded))
            if use_cuda:
                input_var_loc = input_var_loc.cuda()
                input_var_tim = input_var_tim.cuda()
                target_var = target_var.cuda()
                history_loc = history_loc.cuda()
                history_uid = history_uid.cuda()
                history_tim = history_tim.cuda()

            model.eval()
            scores = model(input_var_loc, input_var_tim, input_lengths, history_loc, history_tim, history_uid, history_count)
            batch_traj_tim = []
            batch_traj_loc = []
            target1 = []
            history = []
            history_loc = []
            history_tim = []
            history_uid = []
            history_count = []


        
        loss = masked_cross_entropy(
        scores.contiguous(), 
        target_var.contiguous(), 
        target_lengths)

        
        acc, TEST_DIC = get_acc(scores, target_var, target_lengths, TEST_DIC)
        

        users_acc.extend(acc)

        
        total_loss.append(loss.data.cpu().numpy())

        if run_batch_number!=0 and run_batch_number%1000 == 0:
            print('Loss:{:.4f} '.format(np.mean(total_loss[run_batch_number-3:run_batch_number], dtype=np.float64)))

    avg_loss = np.mean(total_loss, dtype=np.float64)
    
    acc_all = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] 

    P = []
    R = []
    for i in TEST_DIC.keys():
        if i ==0:
            continue
        if TEST_DIC.get(i)[1] == 0:
            TEST_DIC.get(i)[1] = 1
        if TEST_DIC.get(i)[2] == 0:
            TEST_DIC.get(i)[2] = 1
        Pi = TEST_DIC.get(i)[0] / TEST_DIC.get(i)[1]
        Ri = TEST_DIC.get(i)[0] / TEST_DIC.get(i)[2]
        P.append(Pi)
        R.append(Ri)
    macro_R = np.mean(R)
    macro_P = np.mean(P)
    macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)

    for i in range(len(users_acc)):
        for j in range(len(acc_all)):            
            acc_all[j] += users_acc[i][j]


    for j in range(len(acc_all)):
        acc_all[j] = acc_all[j]/len(users_acc)
    acc_all.extend([macro_P, macro_R, macro_F1])

    return avg_loss, acc_all