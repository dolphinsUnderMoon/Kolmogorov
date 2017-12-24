import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.utils.data as Data
import torchvision.transforms
import matplotlib.pyplot as plt

import os


class Config:
    def __init__(self):
        self.max_epoch = 1
        self.batch_size = 64
        self.learning_rate = 1e-2
        self.time_step = 28  # mnist image height
        self.input_size = 28  # rnn input size, mnist image width
        self.is_mnist_exist = True
        self.show_period = 50
        self.input_channel = 1


rnn_config = Config()

if not os.path.exists("./mnist") or not os.listdir("./mnist"):
    rnn_config.is_mnist_exist = False

train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=not rnn_config.is_mnist_exist
)

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title("%i" % train_data.train_labels[0])
plt.show()

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=rnn_config.batch_size,
                               shuffle=True)

test_data = torchvision.datasets.MNIST(root="./mnist", train=False, transform=torchvision.transforms.ToTensor())
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[: 2000]/255.
test_y = test_data.test_labels.numpy().squeeze()[: 2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=rnn_config.input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.output = nn.Linear(64, 10)

    def forward(self, x):
        rnn_out, (h_n, h_c) = self.rnn(x, None)
        out = self.output(rnn_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=rnn_config.learning_rate)
loss_function = nn.CrossEntropyLoss()

for epoch in range(rnn_config.max_epoch):
    for step, (x, y) in enumerate(train_loader):
        batch_x = Variable(x.view(-1, 28, 28))
        batch_y = Variable(y)

        output = rnn(batch_x)
        loss = loss_function(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % rnn_config.show_period == 0:
            test_output = rnn(test_x)
            test_predict = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(test_predict == test_y) / float(test_y.size)
            print("Epoch: [%d], step: [%d], training loss: [%.4f], testing accuracy: [%.4f]" %
                  (epoch, step, loss.data[0], accuracy))


test_output = rnn(test_x[:10].view(-1, 28, 28))
test_predict = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(test_predict, 'prediction number')
print(test_y[:10], 'real number')
