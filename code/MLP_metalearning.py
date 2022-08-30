import torch
import numpy as np

import torch.nn.functional as F
from torch.utils.data import *
import matplotlib.pyplot as plt


class MLP(torch.nn.Module):
    def __init__(self,input_dimension = 1,output_dimension = 2,hidden_dimension = 60):
        super(MLP, self).__init__()

        self.hidden_dimension = hidden_dimension
        self.fc1 = torch.nn.Linear(input_dimension,hidden_dimension)
        self.fc_meta = []

        for _ in range(10):
            self.fc_meta.append(torch.nn.Linear(input_dimension,hidden_dimension))
        self.fc2 = torch.nn.Linear(hidden_dimension,hidden_dimension)
        self.fc3 = torch.nn.Linear(hidden_dimension,output_dimension)


    def forward(self,x):

        x2_persample = torch.zeros((x.shape[0],self.hidden_dimension,10))
        one_hot,delta,phi = x[:,0:10],x[:,10:11],x[:,11:12]
        #print(one_hot.shape)
        #print(i.shape)
        #print(i)
        x1 = self.fc1(delta)
        for i in range(10):
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
test_sample = 0
for i in range(10):
    data = np.loadtxt('../data/data_sample_'+str(i)+'.txt')
    data = torch.from_numpy(data).type(torch.float)

    data = data[1:len(data)-1]

    #if i!=test_sample:

    num_sample = data.shape[0]
    idx = torch.randperm(num_sample)
    train = int(0.8*num_sample)
    data = data[idx]
    phi_mean = torch.mean(data[:,2])
    phi_mean_vec = torch.ones((num_sample,1))*phi_mean
        #phi_mean_vec = data[:,2:3]

    one_hot_vector = torch.zeros((1,10))
    one_hot_vector[0,i] = 1

    one_hot_matrix = torch.concat([one_hot_vector for _ in range(num_sample)],axis = 0)
    #print(one_hot_matrix)

    if i!=test_sample:
        if train_datax is None:
            train_datax = torch.cat([one_hot_matrix[:train],data[:train,3:4],phi_mean_vec[:train,:]],axis = 1)
            train_datay = data[:train,4:6]
            test_datax = torch.cat([one_hot_matrix[train:],data[train:,3:4],phi_mean_vec[train:,:]],axis = 1)
            test_datay = data[train:,4:6]
        else:
            train_datax = torch.cat([train_datax,torch.cat([one_hot_matrix[:train],data[:train,3:4],phi_mean_vec[:train,:]],axis = 1)],axis = 0)
            train_datay = torch.cat([train_datay, data[:train, 4:6]], axis=0)
            test_datax = torch.cat([test_datax,torch.cat([one_hot_matrix[train:],data[train:,3:4],phi_mean_vec[train:,:]],axis = 1)],axis = 0)
            test_datay = torch.cat([test_datay, data[train:, 4:6]], axis=0)
    else:
        meta_trainx = torch.cat([one_hot_matrix[:train], data[:train, 3:4], phi_mean_vec[:train, :]], axis=1)
        meta_trainy = data[:train, 4:6]
        meta_testx = torch.cat([one_hot_matrix[train:], data[train:, 3:4], phi_mean_vec[train:, :]], axis=1)
        meta_testy = data[train:, 4:6]





print(f'training samples: {train_datax.shape[0]}   test samples: {test_datax.shape[0]}')

ntrain = train_datax.shape[0]
ntest = test_datax.shape[0]


batch_size = 20
train_loader = DataLoader(TensorDataset(train_datax,train_datay),batch_size= batch_size,shuffle=True)
test_loader = DataLoader(TensorDataset(test_datax,test_datay),batch_size= batch_size,shuffle=False)

meta_train_loader = DataLoader(TensorDataset(meta_trainx,meta_trainy),batch_size= 5,shuffle=True)
meta_test_loader = DataLoader(TensorDataset(meta_testx,meta_testy),batch_size= 5,shuffle=False)


## training parameters
learning_rate = 1e-2
weight_decay = 1e-3
num_epochs = 2000
device = torch.device('cpu')

model_filename = './MLP_metalearning.ckpt'
model = MLP(input_dimension=1,output_dimension=2,hidden_dimension=40).to(device)

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

### train the 9 tasks
for epoch in range(0):
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








l1, = plt.plot(train_loss,label='train')
l2, = plt.plot(test_loss,label='test')
plt.legend(handles = [l1,l2], fontsize = 20)
plt.xlabel('Epoch',fontsize = 20)
plt.ylabel('Relative error', fontsize = 20)
plt.show()




print('use the learned model, do meta train')




model.load_state_dict(torch.load(model_filename))
optimizer = torch.optim.Adam(model.fc_meta[test_sample].parameters(),lr=learning_rate,weight_decay=1e-4)

metatrain_loss = []
metatest_loss = []
loss_all_min = 100

ntrain = meta_trainx.shape[0]
ntest = meta_testx.shape[0]

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








l1, = plt.plot(metatrain_loss,label='meta-train')
l2, = plt.plot(metatest_loss,label='meta-test')
plt.legend(handles = [l1,l2], fontsize = 20)
plt.xlabel('Epoch',fontsize = 20)
plt.ylabel('Relative error', fontsize = 20)
plt.show()
