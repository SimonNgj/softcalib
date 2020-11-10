# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 00:01:02 2020

@author: xngu0004
"""

import os
import csv
import time
import numpy as np
#from numpy import save
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
from Utils.tcn import TemporalConvNet
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
#from Utils.keras_utils_actuator import SELayer
#from torchsummary import summary
#from tensorboardX import SummaryWriter
#######################################################

CUDA_ID = 0
OUTPUT_DIM = 1
CUDA_ID = 0
N_EPOCHS = 100
seq_length = 50
BATCH_SIZE = 50
input_dim = 1

#######################################################
# 0: LSTM1 || 1: GRU1 || 2: TCN1 || 3: FCN1 || 4: biLSTM1  || 5: biGRU1
model_ID = 3

#######################################################

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_ID)
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
print("DEVICE = ", DEVICE)

#######################################################

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(2).view(b,c)
        y = self.fc(y)
        y = y.view(b,c,1)
        return x * y.expand_as(x)
    
#######################################################
        
class DiabetesDataset(torchDataset):
    def __init__(self, x_data, y_dataF):
        self.len = x_data.size(0)
        self.x_data = x_data
        self.y_dataF = y_dataF

    def __getitem__(self, index):
        return self.x_data[index], self.y_dataF[index]

    def __len__(self):
        return self.len

# Randomize the indexes
def preprocessing(data):
    random_seed = 1234
    len_data = np.shape(data)[0]
    batchdataindex = range(seq_length, len_data)
    permindex = np.array(batchdataindex)
    rng = np.random.RandomState(random_seed)
    rng.shuffle(permindex)
    return permindex

################## Training data ####################
    
train_file = "./data/hysteresis_v_150_1hz1_training.csv"
sensor_data = []

with open(train_file, "r") as f:
    header = f.readline()
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        sensor_data.append([float(row[2]), float(row[3])])  # Trial, V, Grad, Ref_force


train_seq = preprocessing(sensor_data)
total_batch = np.shape(train_seq)[0] // BATCH_SIZE

sensor_input = np.array(sensor_data)[:, 1]  # V
sensor_output = np.array(sensor_data)[:, 0]  # pos

seq_idxs = np.array([train_seq - n for n in reversed(range(0, seq_length))]).T

seq_x = np.reshape(sensor_input[seq_idxs], [-1, seq_length, input_dim])
seq_y = np.reshape(sensor_output[train_seq], [-1, 1])

################## Training data ####################

test_file = "./data/hysteresis_v_150_1hz1_testing.csv"
sensor_dataT = []

with open(test_file, "r") as f:
    header = f.readline()
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        sensor_dataT.append([float(row[2]), float(row[3])])  # Trial, V, Grad, Ref_force

test_seq = preprocessing(sensor_dataT)
total_batchT = np.shape(test_seq)[0] // BATCH_SIZE

sensor_inputT = np.array(sensor_dataT)[:, 1]  # V
sensor_outputT = np.array(sensor_dataT)[:, 0]  # pos

seq_idxsT = np.array([test_seq - n for n in reversed(range(0, seq_length))]).T

seq_xT = np.reshape(sensor_inputT[seq_idxsT], [-1, seq_length, input_dim])
seq_yT = np.reshape(sensor_outputT[test_seq], [-1, 1])   

###########################################################
class actuatorNET(nn.Module, BaseEstimator, RegressorMixin):
    def __init__(self, n_input: int, n_output: int, 
                 n_lstm_layer=3, n_lstm_hidden=64, 
                 n_KDN_hidden=64, lr=1e-3,
                 n_epochs=10, 
                 batch_size=50, 
                 writer=None):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_lstm_layer = n_lstm_layer
        self.n_lstm_hidden = n_lstm_hidden
        self.n_KDN_hidden = n_KDN_hidden
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.writer = writer
        self.num_channel = 64
        self.num_block = 1
        self.num_class = 3

        self.n_hidden_1 = 64
        self.n_hidden_2 = 64
        self.n_hidden_3 = 64

        self.train_x = None
        self.train_y = None

        self.SEN_GRU = None
        self.SEN_LSTM = None
        self.LOC = None
        self.LOC_TCN = None
        self.LOC_bi = None
        self.convo = None
        self.FORCE = None
        self.FORCE_TCN = None
        self.FORCE_bi = None
        self.tcn = None
        self.se = None
        self.criterionF = None
        self.criterionL = None
        self.optimizer = None
        self.valid_data = False

        self._build_model()
        self.to(DEVICE)

        self._set_optimizer()

    def _build_model(self):
        
        self.SEN_LSTM = nn.LSTM(
            input_size=self.n_input,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.n_lstm_layer,
            dropout=0.5
        )  
        
        self.SEN_LSTM_bi = nn.LSTM(
            input_size=self.n_input,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.n_lstm_layer,
            dropout=0.5,
            bidirectional=True
        ) 
        
        self.SEN_GRU = nn.GRU(
            input_size=self.n_input,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.n_lstm_layer,
            dropout=0.5
        ) 
        
        self.SEN_GRU_bi = nn.GRU(
            input_size=self.n_input,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.n_lstm_layer,
            dropout=0.5,
            bidirectional=True
        ) 
        
        self.tcn = TemporalConvNet(
            num_inputs = self.n_input, 
            num_channels=[self.num_channel]*self.num_block, 
            kernel_size=5, 
            dropout=0
            )
        
        self.convo = nn.Sequential(
                nn.Conv1d(self.n_input, self.num_channel, kernel_size=7, padding=3),
                nn.BatchNorm1d(self.num_channel),
                #nn.Dropout(0.25),
                nn.ReLU(),
          
                nn.Conv1d(self.num_channel, self.num_channel, kernel_size=5, padding=2),
                nn.BatchNorm1d(self.num_channel),
                #nn.Dropout(0.25),
                nn.ReLU(),
                
                nn.Conv1d(self.num_channel, self.num_channel, kernel_size=5, padding=2),
                nn.BatchNorm1d(self.num_channel),
                #nn.Dropout(0.25),
                nn.ReLU(),
                
                #nn.AvgPool1d(kernel_size=self.num_channel)
            )
        
        self.se = SELayer(self.num_channel, 8)
        
        self.FORCE = nn.Sequential(
            nn.Linear(self.n_lstm_hidden + self.n_input, self.n_hidden_1),
            nn.ReLU(),

            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU()
        )
        
        self.FORCE_TCN = nn.Sequential(
            nn.Linear(self.num_channel + self.n_input, self.n_hidden_1),
            nn.ReLU(),

            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU(), 
        )
        
        self.FORCE_bi = nn.Sequential(
            nn.Linear(2*self.n_lstm_hidden + self.n_input, self.n_hidden_1),
            nn.ReLU(),

            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU()
        )
         
        self.FORCE1 =  nn.Sequential(
            nn.Linear(self.n_hidden_2, self.num_class*self.n_hidden_2),
            nn.ReLU()
        )
        
        self.FORCE2 =  nn.Sequential(
            nn.Linear(self.num_class*self.n_hidden_2, self.n_hidden_2),
            nn.ReLU()
        )
        
        self.FORCE3 =  nn.Linear(self.n_hidden_2, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def _set_optimizer(self):
        self.criterionF = nn.MSELoss()
        self.criterionL = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, x):
        if model_ID == 0: # LSTM1
            #print("x: ", x.shape)           
            x1 = x.transpose(1, 0)
            #print("x1: ", x1.shape)
            x2 = x1[:-1,:,:]
            #print("x2: ", x2.shape)
            r, _ = self.SEN_LSTM(x2)
            #print("r1: ", r.shape)
            r = r[-1, :, :]
            #print("r2: ", r.shape)     
            r = torch.cat([r, x[:, -1, :]], 1)
            #print("r3: ", r.shape)
            f0 = self.FORCE(r)
            #print("f0: ", f0.shape)
            f3 = self.FORCE3(f0)
            #print("f3_2: ", f3.shape)
            return f3 
        
        elif model_ID == 1: # GRU1
            #print("x: ", x.shape)           
            x1 = x.transpose(1, 0)
            #print("x1: ", x1.shape)
            x2 = x1[:-1,:,:]
            #print("x2: ", x2.shape)
            r, _ = self.SEN_GRU(x2)
            #print("r1: ", r.shape)
            r = r[-1, :, :]
            #print("r2: ", r.shape)
            r = torch.cat([r, x[:, -1, :]], 1)
            #print("r3: ", r.shape)
            f0 = self.FORCE(r)
            #print("f0: ", f0.shape)
            f3 = self.FORCE3(f0)
            #print("f3_2: ", f3.shape)
            return f3 
        
        elif model_ID == 2: # TCN1
            #print("x: ", x.shape)
            #print("xT: ", x.transpose(1,2).shape)
            r = self.tcn(x.transpose(1,2))
            #print("r1: ", r.shape)
            r = r[:, :, -1]
            #print("r2: ", r.shape)
            r = torch.cat([r, x[:, -1, :]], 1)
            #print("r3: ", r.shape)
            l = self.FORCE_TCN(r)
            f3 = self.FORCE3(l)
            #print("f3_2: ", f3.shape)
            return f3
        
        elif model_ID == 3: # FCN1
            #print("x: ", x.shape)
            x1 = x.transpose(1,2)
            #print("x1: ", x1.shape)
            x2 = x1[:,:,:-1]
            #print("x2: ", x1.shape)
            r = self.convo(x2)
            #print("r1: ", r.shape)
            r = self.se(r)
            r = r.mean(2)
            #print("r2: ", r.shape)
            r = torch.cat([r, x[:, -1, :]], 1)
            #print("r3: ", r.shape)
            l = self.FORCE_TCN(r)
            #print("l: ", l.shape)
            f3 = self.FORCE3(l)
            #print("f3_2: ", f3.shape)
            return f3
        
        if model_ID == 4: # biLSTM1
            #print("x: ", x.shape)
            x1 = x.transpose(1,2)
            #print("x1: ", x1.shape)
            x2 = x1[:,:,:-1]
            #print("x2: ", x1.shape)
            r, _ = self.SEN_LSTM_bi(x2)
            #print("r1: ", r.shape)
            r = r[-1, :, :]
            #print("r2: ", r.shape)     
            r = torch.cat([r, x[:, -1, :]], 1)
            #print("r3: ", r.shape)
            l = self.FORCE_bi(r)
            #print("l: ", l.shape)
            f3 = self.FORCE3(l)
            #print("f3_2: ", f3.shape)
            return f3 
        
        elif model_ID == 5: # biGRU1
            #print("x: ", x.shape)
            #print("xt: ", x.transpose(1,2).shape)
            r, _ = self.SEN_GRU_bi(x.transpose(1, 0))
            #print("r1: ", r.shape)
            r = r[-1, :, :]
            #print("r2: ", r.shape)
            r = torch.cat([r, x[:, -1, :]], 1)
            #print("r3: ", r.shape)
            l = self.FORCE_bi(r)
            #print("l: ", l.shape)
            f3 = self.FORCE3(l)
            #print("f3_2: ", f3.shape)
            return f3 
        
###################################################################

    def fit(self, X: np.ndarray, yF: np.ndarray, X_valid=None, yF_valid=None):

        # preprocessing
        self.train_x = torch.from_numpy(seq_x).type(torch.float).to(DEVICE)
        self.train_yF = torch.from_numpy(seq_y).type(torch.float).to(DEVICE)
        
        train_dataset_loader = torchDataLoader(
            dataset=DiabetesDataset(self.train_x, self.train_yF),
                                               batch_size=self.batch_size,
                                               shuffle=True, drop_last=False)

        if (X_valid is not None) and (yF_valid is not None):
            self.valid_data = True
        else:
            pass
        
        for epoch in tqdm(range(self.n_epochs)):
            for i, (x, yF) in enumerate(train_dataset_loader, 0):
                self.train()
                self.optimizer.zero_grad()
                y_predF = self(x)

                lossF = self.criterionF(y_predF, yF)

                loss = lossF
                loss.backward()

                # optimizer mode
                if self.valid_data:
                    pass
                else:
                    self.optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar('tensorBoard/train_loss', loss.item(), epoch )
        
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            yF = self(X)
        return yF

###########################################################

if __name__ == '__main__':
    
    for i in range(0, 5):
        #writer = SummaryWriter()  
        #loader = DataLoader()

        softsennet = actuatorNET(input_dim, OUTPUT_DIM, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, writer=None)
    
        #train_x, train_y = loader.getStandardTrainDataSet()
        #summary(softsennet, (INPUT_LEN, INPUT_DIM))
               
        start_train = time.time()
        softsennet.fit(seq_x, seq_y)
        end_train = time.time()
    
        x_data_torch = torch.from_numpy(seq_xT).type(torch.float).to(DEVICE)
        predF = softsennet.predict(x_data_torch)
        end_test = time.time()
    
        time_train = end_train-start_train
        time_test = end_test-end_train
        
        ############################ Print out the result ###############################
        
        print('')
        print('Total time train = ', time_train)
        print('Total time test = ', time_test)
    
        predF1 = predF.cpu().numpy()

        MAE = metrics.mean_absolute_error(predF1, seq_yT)
        MSE = metrics.mean_squared_error(predF1, seq_yT)
        RMSE = np.sqrt(MSE)
        NRMSE = 100*RMSE / (np.max(seq_yT) - np.min(seq_yT))
        
        print('MAE (kPa) = ', MAE)
        print('MSE (kPa) = ', MSE)
        print('RMSE (kPa) = ', RMSE)
        print('NRMSE (%) = ', NRMSE)     

        #writer.close()    
        #save('./savedResults/pT_'+str(model_ID)+'M'+str(i)+'_predF1.npy', predF1)
        #save('./savedResults/pT_'+str(model_ID)+'M'+str(i)+'_seqxT.npy', seq_xT)
        #save('./savedResults/pT_'+str(model_ID)+'M'+str(i)+'_seqyT.npy', seq_yT)
        
        f = open('./savedResultsC/pT_'+str(model_ID)+'M_performance.txt', 'a+')
        f.write('\n-------Trial ' + str(i) + '---------')
        f.write('\nTotal time train = ' + str(time_train))
        f.write('\nTotal time test = ' + str(time_test))
        f.write('\nMAE (kPa) = ' + str(MAE))
        f.write('\nMSE (kPa) = ' + str(MSE))
        f.write('\nRMSE F (kPa) = ' + str(RMSE))
        f.write('\nNRMSE (%) = ' + str(NRMSE))
        f.write('\n----------------o0o----------------\n')
        f.close()        
        
        ########################## Draw example result #################################
        abc = np.argsort(test_seq[:]) #0->41, 5410->5451
        wpredF1 = predF1[abc[:]]
    
        aF = np.array(sensor_dataT)[seq_length:, (0)] # Force
        aV = np.array(sensor_dataT)[seq_length:, (1)] # Voltage

        fig = plt.figure()
        plt.plot(aF[:], aV[:], 'r')
        plt.plot(wpredF1[:], aV[:], 'b--')
        plt.xlabel('Output (micrometers)')
        plt.ylabel('Input (V)')
        plt.legend(('Actual', 'Prediction'), loc='lower right')
        plt.grid()
        ###################### End of Draw example result ##############################