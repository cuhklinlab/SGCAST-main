import sys
import time
import os
import torch
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from joblib import Parallel, delayed
#import shutil

last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    """Progress Bar for display
    """
    def _format_time(seconds):
        days = int(seconds / 3600/24)
        seconds = seconds - days*3600*24
        hours = int(seconds / 3600)
        seconds = seconds - hours*3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes*60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds*1000)

        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f

    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
    TOTAL_BAR_LENGTH = 30.
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()    # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('    Step: %s' % _format_time(step_time))
    L.append(' | Tot: %s' % _format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# refer to https://github.com/xternalz/WideResNet-pytorch
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "models/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred=[]
    if issparse(dis):
        dis = dis.A
    else:
        dis = dis
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred

def refine_high( pred, dis, num=6, option = False): #shape="high"
    refined_pred=[]
    neigh_idx = dis.tolil().rows
    if option:
        num_nbs = num
        for i in range(len(pred)):
            neigh = pred[neigh_idx[i]]
            if np.max(np.bincount(neigh)) > num_nbs * 0.5:
                refined_pred.append(np.argmax(np.bincount(neigh)))
            else:
                refined_pred.append(pred[i])

    else:
        for i in range(len(pred)):

            neigh = pred[neigh_idx[i]]
            if np.max(np.bincount(neigh)) > len(neigh)*0.5:
                refined_pred.append(np.argmax(np.bincount(neigh)))
            else:
                refined_pred.append(pred[i])

    return refined_pred

def refine_high_old(sample_id, pred, dis, num=6): #shape="high"
    refined_pred=[]
    if issparse(dis):
        dis = dis.A
    else:
        dis = dis
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    num_nbs=num
    # if shape == "high":
    #     num_nbs=10
    # elif shape=="hexagon":
    #     num_nbs=6
    # elif shape=="square":
    #     num_nbs=4
    # else:
    #     print("Shape not recongized, shape='hexagon' for Visium data, shape='high' for high-resolution data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values(ascending=False)
        # num_nbs = sum(dis_tmp>0)
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (np.max(v_c)>num_nbs/2):  #(v_c.loc[self_pred]<num_nbs/2) and
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred

# def ref(i):
#     index=sample_id[i]
#     dis_tmp=dis_df.loc[index, :].sort_values(ascending=False)
#     nbs=dis_tmp[0:num_nbs+1]
#     nbs_pred=pred.loc[nbs.index, "pred"]
#     self_pred=pred.loc[index, "pred"]
#     v_c=nbs_pred.value_counts()
#     if (np.max(v_c)>num_nbs/2):  #(v_c.loc[self_pred]<num_nbs/2) and
#         refined_pred.append(v_c.idxmax())
#     else:
#         refined_pred.append(self_pred)
#     return refined_pred

def refine_high_parallel(sample_id, pred, dis, shape="high"):
    # refined_pred=[]
    if issparse(dis):
        dis = dis.A
    else:
        dis = dis
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "high":
        num_nbs=8
    elif shape=="hexagon":
        num_nbs=6
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, shape='high' for high-resolution data, 'square' for ST data.")

    def ref(i):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values(ascending=False)
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (np.max(v_c) > num_nbs / 2):  # (v_c.loc[self_pred]<num_nbs/2) and
            return v_c.idxmax()
        else:
            return self_pred

    refined_pred = Parallel(n_jobs=1)(delayed(ref)(i) for i in range(len(sample_id)))
    # for i in range(len(sample_id)):
    #     index=sample_id[i]
    #     dis_tmp=dis_df.loc[index, :].sort_values(ascending=False)
    #     nbs=dis_tmp[0:num_nbs+1]
    #     nbs_pred=pred.loc[nbs.index, "pred"]
    #     self_pred=pred.loc[index, "pred"]
    #     v_c=nbs_pred.value_counts()
    #     if (np.max(v_c)>num_nbs/2):  #(v_c.loc[self_pred]<num_nbs/2) and
    #         refined_pred.append(v_c.idxmax())
    #     else:
    #         refined_pred.append(self_pred)
    return refined_pred



# def process(i):
#     return i * i
# results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
# print(results)

def find_res(adata,target_num, n_neighbors=10,start=0.4, step=0.1, max_run=40):
    res=start
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.louvain(adata,resolution=res)
    y_pred=adata.obs['louvain'].astype(int).to_numpy()
    old_num=len(np.unique(y_pred))
    print("Start at res = ", res, "step = ", step)
    print("Res = ", res, "Num of clusters = ", old_num)
    run=0
    while old_num!=target_num:
        old_sign=1 if (old_num<target_num) else -1
        res=res+step*old_sign
        sc.tl.louvain(adata,resolution=res)
        y_pred=adata.obs['louvain'].astype(int).to_numpy()
        new_num=len(np.unique(y_pred))
        print("Res = ", res+step*old_sign, "Num of clusters = ", new_num)
        if new_num==target_num:
            res=res+step*old_sign
            print("recommended res = ", str(res))
            return res, y_pred
        new_sign=1 if (new_num<target_num) else -1
        if new_sign==old_sign:
            res=res+step*old_sign
            print("Res changed to", res)
            old_num=new_num
        else:
            step=step/2
            print("Step changed to", step)
        if run >max_run:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res, y_pred
        run+=1
    print("recommended res = ", str(res))
    return res #, y_pred
