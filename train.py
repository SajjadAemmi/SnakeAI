import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

data = pd.read_csv('snake_train_dataset.csv')
Y = data[['direction']]
X = data[['w0', 'w1', 'w2', 'w3',
          'a0', 'a01', 'a1', 'a12', 'a2', 'a23', 'a3', 'a30',
          'b0', 'b01', 'b1', 'b12', 'b2', 'b23', 'b3', 'b30',
          ]]

X = np.array(X, dtype=np.float32)
Y = np.array(Y).flatten()
N = X.shape[0]

X = torch.tensor(X)
Y = torch.LongTensor(Y)

train = torch.utils.data.TensorDataset(X, Y)
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(20, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model = model.to(device)

epochs = 8
learning_rate = 1e-4
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(1, epochs + 1):
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), position=0, leave=True):
        data = data.to(device) 
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        correct += (output.argmax(dim=1) == target).float().sum()

    accuracy = 100 * correct / N
    
    print('Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()), "Accuracy = {}".format(accuracy))
    

torch.save(model.state_dict(), "snake.pt")