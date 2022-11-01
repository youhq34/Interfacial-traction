import torch
import numpy as np

import torch.nn.functional as F
from torch.utils.data import *
import matplotlib.pyplot as plt
import pandas as pd

class MLP(torch.nn.Module):
    def __init__(self,input_dimension = 1,output_dimension = 2,hidden_dimension = 60):
        super(MLP, self).__init__()

        self.hidden_dimension = hidden_dimension
        self.fc1 = torch.nn.Linear(input_dimension,hidden_dimension)
        self.fc_meta = []

        for _ in range(8):
            self.fc_meta.append(torch.nn.Linear(input_dimension,hidden_dimension))
        self.fc2 = torch.nn.Linear(hidden_dimension,hidden_dimension)
        self.fc3 = torch.nn.Linear(hidden_dimension,output_dimension)


    def forward(self,x):

        x2_persample = torch.zeros((x.shape[0],self.hidden_dimension,8))
        one_hot,delta,phi = x[:,0:8],x[:,8:9],x[:,9:10]
        #print(one_hot.shape)
        #print(i.shape)
        #print(i)
        x1 = self.fc1(delta)
        for i in range(8):
            x2_persample[:,:,i] = self.fc_meta[i](phi)
        #print(x2_persample.shape)
        #print(one_hot.unsqueeze(2))
        x2 = torch.matmul(x2_persample,one_hot.unsqueeze(2)).squeeze(2)
        #x3 = torch.cat([x1,x2],axis=1)
        x3 = x1+x2
        x3 = F.relu(x3)
        x3 = self.fc2(x3)
        x3 = F.relu(x3)
        x3 = self.fc3(x3)
        return x3


## read data, input = (num_samples, 2)  |delta| phi
## output_data = (num_sample, 2) sigma_n sigma_t

## in-distribution setting
train_datax = None
train_datay = None
test_datax = None
test_datay = None


#indistribution = 1 ## 1 for in-distribution, 0 for out-of-distribution
test_sample = 7
for i in range(8):
    #data = pd.read_excel('../data/TSR_data/case'+str(i)+'.')
    #data = np.array(data)
    data = np.loadtxt('./data_sample_'+str(i)+'.txt')
    data = torch.from_numpy(data).type(torch.float)

    data = data[1:len(data)-1]

    num_sample = data.shape[0]
    idx = torch.randperm(num_sample)
    train = int(0.8 * num_sample)
    data = data[idx]


    one_hot_vector = torch.zeros((1, 8))
    one_hot_vector[0, i] = 1

    one_hot_matrix = torch.concat([one_hot_vector for _ in range(num_sample)], axis=0)

    if i != test_sample:
        if train_datax is None:
            train_datax = torch.cat([one_hot_matrix[:train], data[:train, 0:2]], axis=1)
            train_datay = data[:train, 2:4]
            test_datax = torch.cat([one_hot_matrix[train:], data[train:, 0:2]], axis=1)
            test_datay = data[train:, 2:4]
        else:
            train_datax = torch.cat([train_datax, torch.cat([one_hot_matrix[:train], data[:train, 0:2]], axis=1)],axis=0)
            train_datay = torch.cat([train_datay, data[:train, 2:4]], axis=0)
            test_datax = torch.cat([test_datax, torch.cat([one_hot_matrix[train:], data[train:, 0:2]], axis=1)],axis=0)
            test_datay = torch.cat([test_datay, data[train:, 2:4]], axis=0)
    else:
        meta_trainx = torch.cat([one_hot_matrix[:train], data[:train, 0:2]], axis=1)
        meta_trainy = data[:train, 2:4]
        meta_testx = torch.cat([one_hot_matrix[train:], data[train:, 0:2]], axis=1)
        meta_testy = data[train:, 2:4]




print(f'training samples: {train_datax.shape[0]}   test samples: {test_datax.shape[0]}')

ntrain = train_datax.shape[0]
ntest = test_datax.shape[0]


batch_size = 10
train_loader = DataLoader(TensorDataset(train_datax,train_datay),batch_size= batch_size,shuffle=True)
test_loader = DataLoader(TensorDataset(test_datax,test_datay),batch_size= batch_size,shuffle=False)

meta_train_loader = DataLoader(TensorDataset(meta_trainx,meta_trainy),batch_size= batch_size,shuffle=True)
meta_test_loader = DataLoader(TensorDataset(meta_testx,meta_testy),batch_size= batch_size,shuffle=False)


## training parameters
learning_rate = 1e-2
weight_decay = 1e-4
num_epochs = 2000
scheduler_step = 100
scheduler_gamma = 0.7
device = torch.device('cpu')

model_filename = './MLP_metalearning.ckpt'
model = MLP(input_dimension=1,output_dimension=2,hidden_dimension=60).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

def myloss(output,target):
    diff = torch.pow(output-target,2)
    norm = torch.pow(target,2)

    diff = torch.sqrt(torch.sum(diff,axis=1))
    norm = torch.sqrt(torch.sum(norm,axis=1))

    #print(diff,norm)

    return torch.sum(diff/norm)




train_loss = []
test_loss = []
loss_all_min = 100

### train the 9 tasks
for epoch in range(num_epochs):
    loss_all = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = myloss(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    scheduler.step()

    if loss_all/ntrain < loss_all_min:
        torch.save(model.state_dict(), model_filename)
        loss_all_min = loss_all/ntrain
        with torch.no_grad():
            test_loss_all = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                test_loss_all += myloss(output,y).item()

    train_loss.append(loss_all/ntrain)
    test_loss.append(test_loss_all/ntest)
    print(epoch,loss_all/ntrain,loss_all_min,test_loss_all/ntest)







plt.figure()
l1, = plt.plot(train_loss,label='train')
l2, = plt.plot(test_loss,label='test')
plt.legend(handles = [l1,l2], fontsize = 20)
plt.xlabel('Epoch',fontsize = 20)
plt.ylabel('Relative error', fontsize = 20)
plt.savefig(f'./figures_for_mlp/meta_train_first_phase_{test_sample}.png')


if 1:
    data_forplot = None
    prediction_forplot = None
    with torch.no_grad():
        test_loss_all = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            data_forplot = np.array(y.detach())
            prediction_forplot = np.array(output.detach())

    for i in range(5):
        plt.figure()
        plt.plot([0, data_forplot[i, 0]], [0, data_forplot[i, 1]], color='blue', label='Data')
        plt.plot([0, prediction_forplot[i, 0]], [0, prediction_forplot[i, 1]],
                 color='red', label='Prediction')
        xmag = np.abs(data_forplot[i, 0]) if np.abs(data_forplot[i, 0]) > np.abs(prediction_forplot[i, 0]) else np.abs(prediction_forplot[i, 0])
        ymag = np.abs(data_forplot[i, 1]) if np.abs(data_forplot[i, 1]) > np.abs(prediction_forplot[i, 1]) else np.abs(
            prediction_forplot[i, 1])

        mag = xmag if xmag > ymag else ymag
        plt.xlim([-1 * mag, mag])
        plt.ylim([-1 * mag, mag])
        plt.legend()
        plt.savefig(f'./figures_for_mlp/sample_{i}_meta.png')









print('use the learned model, do meta train')




model.load_state_dict(torch.load(model_filename))
optimizer = torch.optim.Adam(model.fc_meta[test_sample].parameters(),lr=learning_rate,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

metatrain_loss = []
metatest_loss = []
loss_all_min = 100

ntrain = meta_trainx.shape[0]
ntest = meta_testx.shape[0]
print(ntrain,ntest)
meta_filename = './meta_train_model.ckpt'

for epoch in range(num_epochs):
    loss_all = 0
    for x, y in meta_train_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = myloss(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    scheduler.step()

    if loss_all/ntrain < loss_all_min:
        torch.save(model.state_dict(), meta_filename)
        loss_all_min = loss_all/ntrain
        with torch.no_grad():
            test_loss_all = 0
            for x, y in meta_test_loader:
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                test_loss_all += myloss(output,y).item()

    metatrain_loss.append(loss_all/ntrain)
    metatest_loss.append(test_loss_all/ntest)
    print(epoch,loss_all/ntrain,loss_all_min,test_loss_all/ntest)


plt.figure()
l1, = plt.plot(metatrain_loss,label='meta-train')
l2, = plt.plot(metatest_loss,label='meta-test')
plt.legend(handles = [l1,l2], fontsize = 20)
plt.xlabel('Epoch',fontsize = 20)
plt.ylabel('Relative error', fontsize = 20)
plt.savefig(f'./figures_for_mlp/meta_train_second_phase_{test_sample}.png')

Visualize_flag = True


plot_case = f'metatest_on_{test_sample}'



if Visualize_flag:
    data_forplot = None
    prediction_forplot = None
    with torch.no_grad():
        test_loss_all = 0
        for x, y in meta_test_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            data_forplot = np.array(y.detach())
            prediction_forplot = np.array(output.detach())

    for i in range(5):
        plt.figure()
        plt.plot([0, data_forplot[i,0]],[0,data_forplot[i,1]],color = 'blue',label = 'Data')
        plt.plot([0, prediction_forplot[i,0]],[0,prediction_forplot[i,1]],color='red', label='Prediction')
        xmag = np.abs(data_forplot[i, 0]) if np.abs(data_forplot[i, 0]) > np.abs(prediction_forplot[i, 0]) else np.abs(
            prediction_forplot[i, 0])
        ymag = np.abs(data_forplot[i, 1]) if np.abs(data_forplot[i, 1]) > np.abs(prediction_forplot[i, 1]) else np.abs(
            prediction_forplot[i, 1])
        mag = xmag if xmag > ymag else ymag
        plt.xlim([-1 * mag, mag])
        plt.ylim([-1 * mag, mag])
        plt.legend()
        plt.savefig(f'./figures_for_mlp/sample_{i}'+plot_case+'.png')