# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 00:17:20 2020

@author: xngu0004
"""

import os
import csv
import numpy as np
from tqdm import tqdm
import time
#from numpy import save
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as torchDataLoader
import sklearn.metrics as metrics
from Utils.tcn import TemporalConvNet
from sklearn.base import BaseEstimator, RegressorMixin
import torch.nn.functional as F
from torch.utils.data import Dataset as torchDataset
#from data.DataLoaderSS import DiabetesDataset
#from Utils.keras_utils_sensor import loc2array, preprocessing
#from torchsummary import summary
#from tensorboardX import SummaryWriter
#######################################################

CUDA_ID = 0
INPUT_DIM = 2
seq_length = 20
OUTPUT_DIM = 1
CUDA_ID = 0
N_EPOCHS = 100
BATCH_SIZE = 50
num_class = 3

# 0: LSTM1 || 1: GRU1 || 2: TCN1 || 3: FCN1 || 4: FCN-LSTM1  
model_ID = 2

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
    def __init__(self, x_data, y_dataL, y_dataF):
        self.len = x_data.size(0)
        self.x_data = x_data
        self.y_dataL = y_dataL
        self.y_dataF = y_dataF

    def __getitem__(self, index):
        return self.x_data[index], self.y_dataL[index], self.y_dataF[index]

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

# Location to Array format
def loc2array(num_array, idx):
    arr = [0] * num_array
    arr[idx-1] = 1
    return arr

######################### Training dataset ##############################

train_file = "./data/loc_train.csv"
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
total_batch = np.shape(train_seq)[0] // BATCH_SIZE

loc_input = np.array(sensor_data)[:, (1, 2)]  # V, gradV
sensor_loc = np.array(sensor_loc)  # Location of pressure
sensor_output = np.array(sensor_data)[:, 3]  # Ref_Force

seq_idxs = np.array([train_seq - n for n in reversed(range(0, seq_length))]).T

seq_x = np.reshape(loc_input[seq_idxs], [-1, seq_length, INPUT_DIM])
v = np.reshape(np.array(sensor_data)[:, 1][train_seq], [-1, 1])
seq_l = np.reshape(sensor_loc[train_seq], [-1, num_class])
seq_y = np.reshape(sensor_output[train_seq], [-1, 1])

######################### Testing dataset ##############################

test_file = "./data/loc_test.csv"
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
total_batchT = np.shape(test_seq)[0] // BATCH_SIZE

loc_inputT = np.array(sensor_dataT)[:, (1, 2)]  # V, gradV
sensor_locT = np.array(sensor_locT)  # Location of pressure
sensor_outputT = np.array(sensor_dataT)[:, 3]  # Ref_Force

seq_idxsT = np.array([test_seq - n for n in reversed(range(0, seq_length))]).T

seq_xT = np.reshape(loc_inputT[seq_idxsT], [-1, seq_length, INPUT_DIM])
vT = np.reshape(np.array(sensor_dataT)[:, 1][test_seq], [-1, 1])
seq_lT = np.reshape(sensor_locT[test_seq], [-1, num_class])
seq_yT = np.reshape(sensor_outputT[test_seq], [-1, 1])
 
#######################################################    

###########################################################
class sensorNET(nn.Module, BaseEstimator, RegressorMixin):
    def __init__(self, n_input: int, n_output: int, 
                 n_lstm_layer=3, n_lstm_hidden=32, 
                 n_KDN_hidden=32, lr=1e-3,
                 n_epochs=10, 
                 batch_size=50, 
                 writer=None):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_lstm_layer = n_lstm_layer
        self.n_lstm_hidden = n_lstm_hidden
        self.n_KDN_hidden = n_lstm_hidden
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.writer = writer
        self.num_channel = 32
        self.num_block = 2
        self.num_class = 3

        self.n_hidden_1 = n_lstm_hidden
        self.n_hidden_2 = n_lstm_hidden
        self.n_hidden_3 = n_lstm_hidden

        self.train_x = None
        self.train_y = None

        self.SEN_GRU = None
        self.SEN_LSTM = None
        self.LOC = None
        self.LOC_TCN = None
        self.LOC_FCNLSTM = None
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
            dropout=0.25
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
                
                nn.Conv1d(self.num_channel, self.num_channel, kernel_size=3, padding=1),
                nn.BatchNorm1d(self.num_channel),
                #nn.Dropout(0.25),
                nn.ReLU(),
                
                #nn.AvgPool1d(kernel_size=self.num_channel)
            )
        
        self.se = SELayer(self.num_channel, 16)
        
        self.LOC = nn.Sequential(
            nn.Linear(self.n_lstm_hidden + self.n_input, self.n_hidden_1),
            nn.ReLU(),

            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU()
        )
        self.LOC_bi = nn.Sequential(
            nn.Linear(2*self.n_lstm_hidden + self.n_input, self.n_hidden_1),
            nn.ReLU(),

            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU()
        )
        self.LOC_lo = nn.Linear(self.n_hidden_2, self.num_class)
        
        self.LOC_TCN = nn.Sequential(
            nn.Linear(self.num_channel + self.n_input, self.n_hidden_1),
            nn.ReLU(),

            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU(),
            
            nn.Linear(self.n_hidden_2, self.n_hidden_2),
        )
        
        self.LOC_FCNLSTM = nn.Sequential(
            nn.Linear(self.n_lstm_hidden + self.n_input + self.num_channel, self.n_hidden_1),
            nn.ReLU(),

            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU()
        )
        
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
        self.AFORCE3 =  nn.Linear(self.n_hidden_2+ self.n_input, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def _set_optimizer(self):
        self.criterionF = nn.MSELoss()
        self.criterionL = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, x):
        if model_ID == 0: # LSTM1        
            x1 = x.transpose(1, 0)
            x2 = x1[:-1,:,:]
            r, _ = self.SEN_LSTM(x2)
            r = r[-1, :, :]   
            r = torch.cat([r, x[:, -1, :]], 1)
            l = self.LOC(r)
            l0 = F.softmax(self.LOC_lo(l), dim=1)
            f0 = self.FORCE1(l)
            f1 = f0.view(-1,self.n_hidden_2, self.num_class)
            l1 = l0.view(l0.shape[0],-1,self.num_class)
            l2 = f1*l1
            l3 = l2.view(l0.shape[0],-1)
            f3 = self.FORCE2(l3)
            f3 = self.FORCE3(f3)
            return l0,f3 
        
        elif model_ID == 1: # GRU1         
            x1 = x.transpose(1, 0)
            x2 = x1[:-1,:,:]
            r, _ = self.SEN_GRU(x2)
            r = r[-1, :, :]
            r = torch.cat([r, x[:, -1, :]], 1)
            l = self.LOC(r)
            l0 = F.softmax(self.LOC_lo(l), dim =1)
            f0 = self.FORCE1(l)
            f1 = f0.view(-1,self.n_hidden_2, self.num_class)
            l1 = l0.view(l0.shape[0],-1,self.num_class)
            l2 = f1*l1
            l3 = l2.view(l0.shape[0],-1)
            f3 = self.FORCE2(l3)
            f3 = self.FORCE3(f3)
            return l0,f3 
        
        elif model_ID == 2: # TCN1
            x1 = x.transpose(1,2)
            x2 = x1[:,:,:-1]
            r = self.tcn(x2)
            r = r[:, :, -1]
            r = torch.cat([r, x[:, -1, :]], 1)
            l = self.LOC_TCN(r)
            l0 = F.softmax(self.LOC_lo(l), dim=1)
            f0 = self.FORCE1(l)
            f1 = f0.view(-1,self.n_hidden_2, self.num_class)
            l1 = l0.view(l0.shape[0],-1,self.num_class)
            l2 = f1*l1
            l3 = l2.view(l0.shape[0],-1)
            f3 = self.FORCE2(l3)
            f3 = self.FORCE3(f3)
            return l0,f3
        
        elif model_ID == 3: # FCN1
            x1 = x.transpose(1,2)
            x2 = x1[:,:,:-1]
            r = self.convo(x2)
            r = self.se(r)
            r = r.mean(2)
            r = torch.cat([r, x[:, -1, :]], 1)
            l = self.LOC_TCN(r)
            l0 = F.softmax(self.LOC_lo(l), dim=1)
            f0 = self.FORCE1(l)
            f1 = f0.view(-1,self.n_hidden_2, self.num_class)
            l1 = l0.view(l0.shape[0],-1,self.num_class)
            l2 = f1*l1
            l3 = l2.view(l0.shape[0],-1)
            f3 = self.FORCE2(l3)
            f3 = self.FORCE3(f3)
            return l0,f3
        
        elif model_ID == 3: # FCN1
            x1 = x.transpose(1,2)
            x2 = x1[:,:,:-1]
            r = self.convo(x2)
            r = self.se(r)
            r = r.mean(2)
            r = torch.cat([r, x[:, -1, :]], 1)
            l = self.LOC_TCN(r)
            l0 = F.softmax(self.LOC_lo(l), dim=1)
            f0 = self.FORCE1(l)
            f1 = f0.view(-1,self.n_hidden_2, self.num_class)
            l1 = l0.view(l0.shape[0],-1,self.num_class)
            l2 = f1*l1
            l3 = l2.view(l0.shape[0],-1)
            f3 = self.FORCE2(l3)
            f3 = self.FORCE3(f3)
            return l0,f3
        
        elif model_ID == 4: # FCN-LSTM1
            x1_FCN = x.transpose(1,2)
            x2_FCN = x1_FCN[:,:,:-1]
            r_FCN = self.convo(x2_FCN)
            r_FCN = self.se(r_FCN)
            r_FCN = r_FCN.mean(2)
            r_FCN = torch.cat([r_FCN, x[:, -1, :]], 1)
            
            x1_LSTM = x.transpose(1, 0)
            x2_LSTM = x1_LSTM[:-1,:,:]
            r_LSTM, _ = self.SEN_LSTM(x2_LSTM)
            r_LSTM = r_LSTM[-1, :, :]
            r = torch.cat([r_LSTM, r_FCN], 1)
            
            l = self.LOC_FCNLSTM(r)
            l0 = F.softmax(self.LOC_lo(l), dim=1)
            f0 = self.FORCE1(l)
            f1 = f0.view(-1,self.n_hidden_2, self.num_class)
            l1 = l0.view(l0.shape[0],-1,self.num_class)
            l2 = f1*l1
            l3 = l2.view(l0.shape[0],-1)
            f3 = self.FORCE2(l3)
            f3 = self.FORCE3(f3)
            return l0,f3

    def fit(self, X: np.ndarray, yL: np.ndarray, yF: np.ndarray, X_valid=None, yL_valid=None, yF_valid=None):

        # preprocessing
        self.train_x = torch.from_numpy(seq_x).type(torch.float).to(DEVICE)
        self.train_yL = torch.from_numpy(seq_l).type(torch.float).to(DEVICE)
        self.train_yF = torch.from_numpy(seq_y).type(torch.float).to(DEVICE)
        
        train_dataset_loader = torchDataLoader(
            dataset=DiabetesDataset(self.train_x, self.train_yL, self.train_yF),
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               drop_last=False)

        if (X_valid is not None) and (yL_valid is not None) and (yF_valid is not None):
            self.valid_data = True
        else:
            pass
        
        for epoch in tqdm(range(self.n_epochs)):
            for i, (x, yL, yF) in enumerate(train_dataset_loader, 0):
                self.train()
                self.optimizer.zero_grad()
                y_predL, y_predF = self(x)

                yLL = torch.max(yL,1)[1]
                lossL = self.criterionL(y_predL, yLL)
                lossF = self.criterionF(y_predF, yF)

                loss = lossL + lossF
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
            yL, yF = self(X)
        return yL, yF

###########################################################

if __name__ == '__main__':
    
    for i in range(10, 11):
        #writer = SummaryWriter()    
        #loader = DataLoader()

        softsennet = sensorNET(INPUT_DIM, OUTPUT_DIM, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, writer=None)
    
        #train_x, train_y = loader.getStandardTrainDataSet()
        #summary(softsennet, (INPUT_LEN, INPUT_DIM))        
        
        start_train = time.time()
        softsennet.fit(seq_x, seq_l, seq_y)
        end_train = time.time()
    
        x_data_torch = torch.from_numpy(seq_xT).type(torch.float).to(DEVICE)
        predL, predF = softsennet.predict(x_data_torch)
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
        RMSE_kPa = RMSE/(25 * np.pi) * 1000 
        NRMSE = 100*RMSE / (np.max(seq_yT) - np.min(seq_yT))
        
        print('MAE (V) = ', MAE)
        print('MSE (V) = ', MSE)
        print('RMSE (V) = ', RMSE)
        print('RMSE (kPa) = ', RMSE_kPa)      
        print('NRMSE (%) = ', NRMSE) 

        #writer.close()
     
        location_test = np.argmax(seq_lT, axis=1)
        predL1 = predL.cpu().numpy()
        location_pred = np.argmax(predL1, axis=1)
        location_dif = location_test - location_pred 
        loc_error = 100-100*np.sum(location_dif == 0)/location_dif.shape[0]
        print('Location error (%) = ', loc_error)
        
        ############################ Save result to files ###############################
        
        #save('./savedResults/pT_'+str(model_ID)+'M'+str(i)+'_predL1.npy', predL1)
        #save('./savedResults/pT_'+str(model_ID)+'M'+str(i)+'_predF1.npy', predF1)
        #save('./savedResults/pT_'+str(model_ID)+'M'+str(i)+'_seqxT.npy', seq_xT)
        #save('./savedResults/pT_'+str(model_ID)+'M'+str(i)+'_seqyT.npy', seq_yT)
        
        f = open('./savedResultsMAE/pT_'+str(model_ID)+'M_performance.txt', 'a+')
        f.write('\n-------Trial ' + str(i) + '---------')       
        f.write('\nMAE (V) = ' + str(MAE))
        f.write('\nMSE (V) = ' + str(MSE))
        f.write('\nRMSE F (V) = ' + str(RMSE))
        f.write('\nRMSE F(kPa) = ' + str(RMSE_kPa))
        f.write('\nNRMSE (%) = ' + str(NRMSE))
        f.write('\nLocation error = ' + str(loc_error))
        f.write('\nTotal time train = ' + str(time_train))
        f.write('\nTotal time test = ' + str(time_test))
        f.write('\n----------------o0o----------------\n')
        f.close()       
        
        ########################## Draw example result #################################
        abc = np.argsort(test_seq[:])
        wpredF1 = predF1[abc[:]]

        a = 1025 - seq_length 
        b = a + 15
    
        aF = np.array(sensor_dataT)[:, (3)] # Force
        aV = np.array(sensor_dataT)[:, (1)] # Voltage

        fig = plt.figure()
        plt.plot(-aF[a+seq_length:b+seq_length], aV[a+seq_length:b+seq_length], 'r')
        plt.plot(-wpredF1[a:b], aV[a+seq_length:b+seq_length], 'b--')
        plt.xlabel('Pressure (kPa)')
        plt.ylabel('Response (V)')
        plt.legend(('Actual', 'Prediction'), loc='lower right')
        plt.grid()
        ###################### End of Draw example result ##############################   
