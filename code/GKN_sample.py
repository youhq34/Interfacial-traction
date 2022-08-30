import torch
import numpy as np

import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from utilities import *
from nn_conv import NNConv_old
import matplotlib.pyplot as plt

from timeit import default_timer


class KernelNN(torch.nn.Module):
    def __init__(self, width_node, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width_node)

        kernel = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width_node, width_node, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width_node, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = self.conv1(x, edge_index, edge_attr)
            if k != self.depth - 1:
                x = F.relu(x)

        x = self.fc2(x)
        return x

## load data from 10 samples

## the input is (deltan,deltat,phi_mean) for each points
## (deltan_i,deltat_i,phi_mean_i,deltan_j,deltat_j,phi_mean_j) for each edge

print('start preprocessing data')

Train_data = []

Test_data = []

ntrain = 9
ntest = 1
test_sample = 5  ## sample No.6 is set as the test set for now

radius = 0.25
for i in range(10):
    data = np.loadtxt('../data/data_sample_'+str(i)+'.txt')
    data = torch.from_numpy(data).type(torch.float)
    data = data[1:len(data)-1]

    phi = data[:,2]
    phi_mean = torch.mean(phi)
    phi_mean_tensor = torch.ones((phi.shape[0]))*phi_mean
    meshgenerator = RandomMesh_fromfile(data[:,0:2])

    print(f'sample: {i}, phi mean: {phi_mean}')

    grid = meshgenerator.get_grid()
    edge_index = meshgenerator.ball_connectivity(radius)
    edge_attr = meshgenerator.attributes(theta=phi_mean_tensor)

    print(f'sample: {i}, grid shape: {grid.shape}, edge shape: {edge_index.shape}, edge_attr shape: {edge_attr.shape}')



    if i == test_sample:
        Test_data.append(Data(x=torch.cat([grid, data[:,2].reshape(-1, 1),
                                        ], dim=1),
                                  y=data[:,4:],edge_index=edge_index, edge_attr=edge_attr
                                  ))
    else:
        Train_data.append(Data(x=torch.cat([grid, data[:, 2].reshape(-1, 1),
                                           ], dim=1),
                              y=data[:, 4:], edge_index=edge_index, edge_attr=edge_attr
                              ))

width = 32
ker_width = 32
depth = 6
edge_features = 6
node_features = 3

epochs = 500
learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.5

batch_size = 2


train_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False)

device = torch.device('cpu')

model = KernelNN(width, ker_width, depth, edge_features, in_width=node_features).cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
ttrain = np.zeros((epochs,))
ttest = np.zeros((epochs,))
model.train()
for ep in range(epochs):
    t1 = default_timer()
    train_mse = 0.0
    train_l2 = 0.0
    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1, 1))
        mse.backward()

        l2 = myloss(out.view(out.shape[0], -1),batch.y.view(batch.y.shape[0], -1))
        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    t2 = default_timer()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            output_test = out.detach().cpu().numpy()
            batchy_data = batch.y.detach().cpu().numpy()

            out = out.view(out.shape[0],-1)
            test_l2 += myloss(out, batch.y.view(batch.y.shape[0], -1)).item()
            # test_l2 += myloss(out.view(batch_size2,-1), y_normalizer.encode(batch.y.view(batch_size2, -1))).item()

    ttrain[ep] = train_l2 / ntrain
    ttest[ep] = test_l2 / ntest

    print(ep, t2 - t1, train_mse / len(train_loader), train_l2 / (ntrain), test_l2 / ntest)

l1, = plt.plot(output_test[:,0],output_test[:,1],'o', label = 'Prediction')
l2, = plt.plot(batchy_data[:,0],batchy_data[:,1],'o',label = 'Data')
plt.legend(handles = [l1,l2])
plt.show()

