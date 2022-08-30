import torch
import numpy as np

import torch.nn.functional as F
from torch.utils.data import *
import matplotlib.pyplot as plt


class MLP(torch.nn.Module):
    def __init__(self,input_dimension = 2,output_dimension = 2,hidden_dimension = 40):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(input_dimension,hidden_dimension)
        self.fc2 = torch.nn.Linear(hidden_dimension,hidden_dimension)
        self.fc3 = torch.nn.Linear(hidden_dimension,output_dimension)


    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


## read data, input = (num_samples, 2)  |delta| phi
## output_data = (num_sample, 2) sigma_n sigma_t

## in-distribution setting
train_datax = None
train_datay = None
test_datax = None
test_datay = None

indistribution = 0 ## 1 for in-distribution, 0 for out-of-distribution
test_sample = 5
for i in range(10):
    data = np.loadtxt('../data/data_sample_'+str(i)+'.txt')
    data = torch.from_numpy(data).type(torch.float)

    data = data[1:len(data)-1]

    if indistribution:

        num_sample = data.shape[0]
        idx = torch.randperm(num_sample)
        train = int(0.8*num_sample)
        data = data[idx]

        phi_mean = torch.mean(data[:,2])
        #phi_mean_vec = torch.ones((num_sample,1))*phi_mean
        phi_mean_vec = data[:,2:3]
        if train_datax is None:
            train_datax = torch.cat([data[:train,3:4],phi_mean_vec[:train,:]],axis = 1)
            train_datay = data[:train,4:6]
            test_datax = torch.cat([data[train:,3:4],phi_mean_vec[train:,:]],axis = 1)
            test_datay = data[train:,4:6]
        else:
            train_datax = torch.cat([train_datax,torch.cat([data[:train,3:4],phi_mean_vec[:train,:]],axis = 1)],axis = 0)
            train_datay = torch.cat([train_datay, data[:train, 4:6]], axis=0)
            test_datax = torch.cat([test_datax,torch.cat([data[train:,3:4],phi_mean_vec[train:,:]],axis = 1)],axis = 0)
            test_datay = torch.cat([test_datay, data[train:, 4:6]], axis=0)
    else:
        ## out-of-distribution
        num_sample = data.shape[0]
        phi_mean = torch.mean(data[:, 2])
        phi_mean_vec = torch.ones((num_sample,1))*phi_mean

        if i == test_sample:
            test_datax = torch.cat([data[:,3:4],phi_mean_vec],axis = 1)
            test_datay = data[:,4:6]
        else:
            if train_datax is None:
                train_datax = torch.cat([data[:,3:4],phi_mean_vec],axis = 1)
                train_datay = data[:,4:6]
            else:
                train_datax = torch.cat([train_datax,torch.cat([data[:,3:4],phi_mean_vec],axis = 1)],axis = 0)
                train_datay = torch.cat([train_datay,data[:,4:6]],axis = 0)


print(f'training samples: {train_datax.shape[0]}   test samples: {test_datax.shape[0]}')

ntrain = train_datax.shape[0]
ntest = test_datax.shape[0]


batch_size = 20
train_loader = DataLoader(TensorDataset(train_datax,train_datay),batch_size= batch_size,shuffle=True)
test_loader = DataLoader(TensorDataset(test_datax,test_datay),batch_size= batch_size,shuffle=True)

## training parameters
learning_rate = 1e-2
weight_decay = 1e-3
num_epochs = 2000
device = torch.device('cpu')

model = MLP(input_dimension=2,output_dimension=2,hidden_dimension=60).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

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

    if loss_all/ntrain < loss_all_min:
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

l1, = plt.plot(train_loss,label='train')
l2, = plt.plot(test_loss,label='test')
plt.legend(handles = [l1,l2], fontsize = 20)
plt.xlabel('Epoch',fontsize = 20)
plt.ylabel('Relative error', fontsize = 20)
plt.show()






