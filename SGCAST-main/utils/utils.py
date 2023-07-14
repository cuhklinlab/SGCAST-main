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

# refer to SpaGCN, use pd.DataFrame to do refinement
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

# pd.DataFrame is highly memory-consuming, therefore for high-resolution data, we use np.bincount to do refinement.
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

