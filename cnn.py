import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE

import os


class Config:
    def __init__(self):
        self.max_epoch = 1
        self.batch_size = 50
        self.learning_rate = 1e-3
        self.is_mnist_exist = True
        self.input_size = 28  # 28 * 28, mnist image size
        self.show_period = 50
        self.input_channel = 1


cnn_config = Config()

if not os.path.exists("./mnist") or not os.listdir("./mnist"):
    cnn_config.is_mnist_exist = False

train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=not cnn_config.is_mnist_exist
)

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title("%i" % train_data.train_labels[0])
plt.show()

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=cnn_config.batch_size,
                               shuffle=True)

test_data = torchvision.datasets.MNIST(root="./mnist", train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[: 2000]/255.
test_y = test_data.test_labels[: 2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=cnn_config.input_channel,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.output = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.view(x.size(0), -1)
        output = self.output(x)
        return output, x


cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=cnn_config.learning_rate)
loss_function = nn.CrossEntropyLoss()


def plot_with_labels(low_dim_weights, labels):
    plt.cla()
    X, Y = low_dim_weights[:, 0], low_dim_weights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("Visualize the last layer")
    plt.show()
    plt.pause(0.01)


plt.ion()

for epoch in range(cnn_config.max_epoch):
    for step, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)

        train_output = cnn(batch_x)[0]
        loss = loss_function(train_output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cnn_config.show_period == 0:
            test_output, last_layer = cnn(test_x)
            test_predict = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(test_predict == test_y) / float(test_y.size(0))

            print("Epoch: [%d], step: [%d], training loss: [%.4f], testing accuracy: [%.4f]" %
                  (epoch, step, loss.data[0], accuracy))

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embeddings = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            labels = test_y.numpy()[:plot_only]
            plot_with_labels(low_dim_embeddings, labels=labels)

plt.ioff()


test_output, _ = cnn(test_x[:10])
test_predict = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(test_predict, "predict number")
print(test_y[:10].numpy(), 'real number')
