import os, sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model

#Note: this code must be run using tensorflow 1.4.0

class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, i, y = [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            i.append(t[2])
            y.append(t[3])
        return self.i, (u, hist, i, y)

def test(sess, model, test_set):
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    arr = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, _, _ = model.test(sess, uij)
        for index in range(len(score)):
            if label[index] > 0:
                arr.append([0, 1, score[index]])
            elif label[index] == 0:
                arr.append([1, 0, score[index]])
    arr = sorted(arr, key=lambda d:d[2])
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def hit_rate(sess, model, test_set):
    hit, arr = [], []
    userid = list(set([x[0] for x in test_set]))
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, _, _ = model.test(sess, uij)
        for index in range(len(score)):
            if score[index] > 0.5:
                arr.append([label[index], 1, user[index]])
            else:
                arr.append([label[index], 0, user[index]])
    for user in userid:
        arr_user = [x for x in arr if x[2]==user and x[1]==1]
        hit.append(sum([x[0] for x in arr_user])/len(arr_user))
    return np.mean(hit)

def coverage(sess, model, test_set):
    rec_item = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, item, _ = model.test(sess, uij)
        for index in range(len(score)):
            if score[index] > 0.5:
                rec_item.append(item[index])
    return len(set(rec_item)) / len(itemid)

def unexpectedness(sess, model, test_set):
    unexp_list = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, item, unexp = model.test(sess, uij)
        for index in range(len(score)):
            unexp_list.append(unexp[index])
    return np.mean(unexp_list)

random.seed(625)
np.random.seed(625)
tf.set_random_seed(625)
batch_size = 32

data = pd.read_csv('test.txt', names=['utdid','vdo_id','click','hour'])
user_id = data[['utdid']].drop_duplicates().reindex()
user_id['user_id'] = np.arange(len(user_id))
data = pd.merge(data, user_id, on=['utdid'], how='left')
item_id = data[['vdo_id']].drop_duplicates().reindex()
item_id['video_id'] = np.arange(len(item_id))
data = pd.merge(data, item_id, on=['vdo_id'], how='left')
data = data[['user_id','video_id','click','hour']]
userid = list(set(data['user_id']))
itemid = list(set(data['video_id']))
user_count = len(userid)
item_count = len(itemid)

validate = 4 * len(data) // 5
train_data = data.loc[:validate,]
test_data = data.loc[validate:,]
train_set, test_set = [], []

for user in userid:
    train_user = train_data.loc[train_data['user_id']==user]
    train_user = train_user.sort_values(['hour'])
    length = len(train_user)
    train_user.index = range(length)
    if length > 10:
        for i in range(length-10):
            train_set.append((train_user.loc[i+9,'user_id'], list(train_user.loc[i:i+9,'video_id']), train_user.loc[i+9,'video_id'], float(train_user.loc[i+9,'click'])))
    test_user = test_data.loc[test_data['user_id']==user]
    test_user = test_user.sort_values(['hour'])
    length = len(test_user)
    test_user.index = range(length)
    if length > 10:
        for i in range(length-10):
            test_set.append((test_user.loc[i+9,'user_id'], list(test_user.loc[i:i+9,'video_id']), test_user.loc[i+9,'video_id'], float(test_user.loc[i+9,'click'])))
random.shuffle(train_set)
random.shuffle(test_set)
train_set = train_set[:len(train_set)//batch_size*batch_size]
test_set = test_set[:len(test_set)//batch_size*batch_size]

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Model(user_count, item_count, batch_size)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    print('test_auc: %.4f' % test(sess, model, test_set))
    sys.stdout.flush()
    lr = 1
    start_time = time.time()
    last_auc = 0.0
    
    for _ in range(1000):
        random.shuffle(train_set)
        epoch_size = round(len(train_set) / batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, batch_size):
            loss = model.train(sess, uij, lr)
            loss_sum += loss        
            if model.global_step.eval() % 100 == 0:
                auc = test(sess, model, test_set)
                train_auc = test(sess, model, train_set)
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f\tTrain_AUC: %.4F' %
                      (model.global_epoch_step.eval(), model.global_step.eval(),loss_sum / 1000, auc, train_auc))
                sys.stdout.flush()
                loss_sum = 0.0            
        print('Epoch %d DONE\tCost time: %.2f' % 
              (model.global_epoch_step.eval(), time.time()-start_time))
        if abs(train_auc - last_auc) < 0.001:
            lr = lr / 2
        last_auc = train_auc
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
        hit = hit_rate(sess, model, test_set)
        cov = coverage(sess, model, test_set)
        unexp = unexpectedness(sess, model, test_set)
        print('Epoch %d Eval_Hit_Rate: %.4f' % (model.global_epoch_step.eval(), hit))
        print('Epoch %d Eval_Coverage: %.4f' % (model.global_epoch_step.eval(), cov))
        print('Epoch %d Eval_Unexpectedness: %.4f' % (model.global_epoch_step.eval(), unexp))