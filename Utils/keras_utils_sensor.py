# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:32:30 2020

@author: xngu0004
"""

import csv
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, Dense, multiply

#######################################################
input_dim = 2
seq_length = 40
num_class = 3

batch_size = 50
total_epoch = 100

random_seed = 123
#######################################################
# Calculate RMSE
def cal_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#######################################################
# Array format to Location
def array2loc(in_arr):
    location = 0
    for i in range(num_class):
        if in_arr[i] == 1:
            location = i + 1
            break
    return location

#######################################################
# Location to Array format
def loc2array(num_array, idx):
    arr = [0] * num_array
    arr[idx-1] = 1
    return arr

#######################################################
# Randomize the indexes
def preprocessing(data):
    len_data = np.shape(data)[0]
    batchdataindex = range(seq_length, len_data)
    permindex = np.array(batchdataindex)
    rng = np.random.RandomState(random_seed)
    rng.shuffle(permindex)
    return permindex

#######################################################
# Squeeze and excitation block
def squeeze_excite_block(input):
    filters = input.shape[-1] 
    seb = GlobalAveragePooling1D()(input)
    seb = Reshape((1, filters))(seb)
    seb = Dense(filters // 8,  activation='relu', kernel_initializer='he_normal', use_bias=False)(seb)
    seb = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(seb)
    seb = multiply([input, seb])
    return seb

#######################################################
# Restart learning rate
class LRrestart(Callback):
    def __init__(self, min_lr, max_lr, steps_per_epoch, lr_decay=1, cycle_length=10, mult_factor=2):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.batch_since_restart = 0
        self.next_restart = cycle_length
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = cycle_length
        self.mult_factor = mult_factor
        self.history = {}
    def clr(self):
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr
    def on_train_begin(self, logs={}):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)
    def on_batch_end(self, batch, logs={}):
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())
    def on_epoch_end(self, epoch, logs={}):
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weights)
        
#######################################################
# Training the model
def getTrainData(train_file_link):
    train_file = train_file_link
    sensor_data = []
    sensor_loc = []

    with open(train_file, "r") as f:
        header = f.readline()
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            x_location = int(row[1]) // 15 + 2
            sensor_data.append([int(row[0]), float(row[5]), float(row[6]), float(row[7])])  # Trial, V, Grad, Ref_force
            sensor_loc.append(loc2array(num_class, x_location))

    train_seq = preprocessing(sensor_data)
    total_batch = np.shape(train_seq)[0] // batch_size

    loc_input = np.array(sensor_data)[:, (1, 2)]  # V, gradV
    sensor_loc = np.array(sensor_loc)  # Location of pressure
    sensor_output = np.array(sensor_data)[:, 3]  # Ref_Force

    seq_idxs = np.array([train_seq - n for n in reversed(range(0, seq_length))]).T

    seq_x = np.reshape(loc_input[seq_idxs], [-1, seq_length, input_dim])
    #v = np.reshape(np.array(sensor_data)[:, 1][train_seq], [-1, 1])
    seq_l = np.reshape(sensor_loc[train_seq], [-1, num_class])
    seq_y = np.reshape(sensor_output[train_seq], [-1, 1])
    
    return seq_x, seq_l, seq_y, total_batch

#######################################################
# Testing the model

def getTestData(test_file_link):
    test_file = test_file_link
    sensor_dataT = []
    sensor_locT = []

    with open(test_file, "r") as f:
        header = f.readline()
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            x_locationT = int(row[1]) // 15 + 2
            sensor_dataT.append([int(row[0]), float(row[5]), float(row[6]), float(row[7])])  # Trial, V, Grad, Ref_force
            sensor_locT.append(loc2array(num_class, x_locationT))

    test_seq = preprocessing(sensor_dataT)
    total_batchT = np.shape(test_seq)[0] // batch_size

    loc_inputT = np.array(sensor_dataT)[:, (1, 2)]  # V, gradV
    sensor_locT = np.array(sensor_locT)  # Location of pressure
    sensor_outputT = np.array(sensor_dataT)[:, 3]  # Ref_Force

    seq_idxsT = np.array([test_seq - n for n in reversed(range(0, seq_length))]).T

    seq_xT = np.reshape(loc_inputT[seq_idxsT], [-1, seq_length, input_dim])
    #vT = np.reshape(np.array(sensor_dataT)[:, 1][test_seq], [-1, 1])
    seq_lT = np.reshape(sensor_locT[test_seq], [-1, num_class])
    seq_yT = np.reshape(sensor_outputT[test_seq], [-1, 1])
    
    return seq_xT, seq_lT, seq_yT, total_batchT
  
#######################################################   
# print out the results
def print_result(options, est_f, ref_f, est_loc, ref_loc, test_seq):
    total_force_truth_results = [[], [], []]
    est_results = [[], [], []]

    each_force_truth_results = [[], [], [], [], []]
    force_est_results = [[], [], [], [], []]

    interval = 50
    for i in range(len(options)):
        x_location = int(options[i][1])-1
        est_location = np.argmax(est_loc[i])
        ref_force = -float(ref_f[i])
        est_force = -float(est_f[i])

        total_force_truth_results[x_location].append(ref_force)
        est_results[x_location].append(est_force)

        ref_force_kPa = ref_force / (25*np.pi) *1000

        each_force_truth_results[int(ref_force_kPa/(interval))].append(x_location)
        force_est_results[int(ref_force_kPa/(interval))].append(est_location)


    # Total Regression Results
    RMSE = rmse(np.array(est_f), np.array(ref_f))
    NRMSE = RMSE / (np.max(est_f) - np.min(est_f))
    RMSE_kPa = RMSE / (25*np.pi) * 1000
    print("== Regression Result ==")
    print("Overall  RMSE: {:.4}	| NRMSE %: {:.3}".format(RMSE_kPa, NRMSE * 100))
    print("=========================================")

    # Each Regression Result
    for c in range(num_class):
        RMSE = rmse(np.array(est_results[c]), np.array(total_force_truth_results[c]))
        NRMSE = RMSE / (np.max(est_results[c]) - np.min(est_results[c]))
        RMSE_kPa = RMSE / (25 * np.pi) * 1000
        print("Loc {} |  RMSE: {:.4}	| NRMSE %: {:.3}".format(c, RMSE_kPa, NRMSE * 100))
    print("")


    TOTAL = 0
    SUCCESS = 1
    accurate_cnt = 0
    accurate_cnts = np.zeros((num_class, 2))
    loc_results = np.zeros((num_class, num_class))
    for i in range(len(est_loc)):
        ref = int(ref_loc[i]) - 1
        loc = np.argmax(est_loc[i])
        loc_results[ref][loc] += 1
        accurate_cnts[ref][TOTAL] += 1  # count total
        if ref == loc:
            accurate_cnts[ref][SUCCESS] += 1  # count success
            accurate_cnt += 1
        # print ("{:5}: REF {} / EST {}".format(int(merge[0]), merge[1], merge[2]))

    # Total Location Result
    print("== Localization Result ==")
    print("Overall {}/{} : {:.2f}%".format(accurate_cnt, len(est_loc), accurate_cnt / len(est_loc) * 100))
    print("=========================================")

    # Each Location Result
    for i in range(num_class):
        print("Loc {} | {:4}/{:4} : {:.2f}%   {:4}|{:4}|{:4}".format(i + 1, int(accurate_cnts[i][SUCCESS]),
                                                                    int(accurate_cnts[i][TOTAL]), round(
                accurate_cnts[i][SUCCESS] / accurate_cnts[i][TOTAL] * 100, 4)
                                                                    , int(loc_results[i][0]), int(loc_results[i][1]),
                                                                    int(loc_results[i][2])))

    #loc_result_cnts = np.zeros((FLAGS.num_class, len(force_truth_results), FLAGS.num_class))
    loc_accurate_cnts = np.zeros((num_class, len(each_force_truth_results), 2))
    for i in range(len(each_force_truth_results)):
        for j in range(len(each_force_truth_results[i])):
            x_location = each_force_truth_results[i][j]
            #loc_result_cnts[x_location][force_est_results[i][j]] += 1
            loc_accurate_cnts[x_location][i][TOTAL] += 1
            if x_location == force_est_results[i][j]:
                loc_accurate_cnts[x_location][i][SUCCESS] +=1

    for r in range(len(each_force_truth_results)):
        loc_overall_success = loc_accurate_cnts[0][r][SUCCESS] + loc_accurate_cnts[1][r][SUCCESS] + loc_accurate_cnts[2][r][SUCCESS]
        loc_overall_total = loc_accurate_cnts[0][r][TOTAL] + loc_accurate_cnts[1][r][TOTAL] +  loc_accurate_cnts[2][r][TOTAL]
        if loc_overall_total != 0:
             loc_overall = loc_overall_success / loc_overall_total * 100
        else: loc_overall = 0
        if loc_accurate_cnts[0][r][TOTAL] != 0 :
            loc1 = loc_accurate_cnts[0][r][SUCCESS] / loc_accurate_cnts[0][r][TOTAL] * 100
        else : loc1 = 0
        if loc_accurate_cnts[1][r][TOTAL] != 0:
            loc2 = loc_accurate_cnts[1][r][SUCCESS] / loc_accurate_cnts[1][r][TOTAL] * 100
        else : loc2 = 0
        if loc_accurate_cnts[2][r][TOTAL] != 0:
            loc3 = loc_accurate_cnts[2][r][SUCCESS] / loc_accurate_cnts[2][r][TOTAL] * 100
        else : loc3 = 0
        print("Range {:2}<=kPa<{:2} | {:3.2f}% | {:3.2f}% | {:3.2f}% | {:3.2f}%".format(r*interval, (r+1)*interval, loc_overall, loc1, loc2, loc3))

#######################################################  