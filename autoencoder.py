import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Config:
    def __init__(self):
        self.max_epoch = 10
        self.batch_size = 64
        self.learning_rate = 5e-3
        self.is_mnist_exist = True
        self.input_size = 28  # 28 * 28, mnist image size
        self.num_test_image = 5
        self.show_period = 100


autoencoder_config = Config()

train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=not autoencoder_config.is_mnist_exist
)

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
plt.title("%i" % train_data.train_labels[2])

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=autoencoder_config.batch_size,
                               shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(autoencoder_config.input_size ** 2, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, autoencoder_config.input_size ** 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=autoencoder_config.learning_rate)
loss_function = nn.MSELoss()

f, a = plt.subplots(2, autoencoder_config.num_test_image, figsize=(5, 2))
plt.ion()

view_data = Variable(train_data.train_data[:autoencoder_config.num_test_image].view(-1, autoencoder_config.input_size ** 2).type(torch.FloatTensor)/255.)
for i in range(autoencoder_config.num_test_image):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (autoencoder_config.input_size, autoencoder_config.input_size)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for epoch in range(autoencoder_config.max_epoch):
    for step, (x, y) in enumerate(train_loader):
        batch_x = Variable(x.view(-1, autoencoder_config.input_size ** 2))
        batch_y = Variable(x.view(-1, autoencoder_config.input_size ** 2))
        batch_label = Variable(y)

        encoded, decoded = autoencoder(batch_x)

        loss = loss_function(decoded, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % autoencoder_config.show_period == 0:
            print("Epoch: [%d], step: [%d], training loss: %.4f" % (epoch, step, loss.data[0]))

            _, decoded_data = autoencoder(view_data)
            for i in range(autoencoder_config.num_test_image):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (autoencoder_config.input_size, autoencoder_config.input_size)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()

view_data = Variable(train_data.train_data[:200].view(-1, autoencoder_config.input_size ** 2).type(torch.FloatTensor)/255.)
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
