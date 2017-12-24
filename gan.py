import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class Config:
    def __init__(self):
        self.batch_size = 64
        self.generator_learning_rate = 1e-4
        self.discriminator_learning_rate = 1e-4
        self.num_noises = 5
        self.num_art_components = 15
        self.paint_points = np.vstack([np.linspace(-1, 1, self.num_art_components) for _ in range(self.batch_size)])
        self.training_max_iterations = 10000
        self.show_period = 50


gan_config = Config()


def target_work():
    a = np.random.uniform(1, 2, size=gan_config.batch_size)[:, np.newaxis]
    targets = a * np.power(gan_config.paint_points, 2) + (a - 1)
    targets = torch.from_numpy(targets).float()
    return Variable(targets)


generator = nn.Sequential(
    nn.Linear(gan_config.num_noises, 128),
    nn.ReLU(),
    nn.Linear(128, gan_config.num_art_components)
)

discriminator = nn.Sequential(
    nn.Linear(gan_config.num_art_components, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=gan_config.discriminator_learning_rate)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=gan_config.generator_learning_rate)

plt.ion()

for step in range(gan_config.training_max_iterations):
    target_paintings = target_work()

    generator_noises = Variable(torch.randn(gan_config.batch_size, gan_config.num_noises))
    generator_output = generator(generator_noises)

    predict_for_target = discriminator(target_paintings)
    predict_for_generator = discriminator(generator_output)

    loss_discriminator = -torch.mean(torch.log(predict_for_target) + torch.log(1. - predict_for_generator))
    loss_generator = torch.mean(torch.log(1. - predict_for_generator))

    generator_optimizer.zero_grad()
    loss_generator.backward(retain_variables=True)
    generator_optimizer.step()

    discriminator_optimizer.zero_grad()
    loss_discriminator.backward()
    discriminator_optimizer.step()

    if step % gan_config.show_period == 0:
        plt.cla()
        plt.plot(gan_config.paint_points[0], generator_output.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(gan_config.paint_points[0], 2 * np.power(gan_config.paint_points[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(gan_config.paint_points[0], 1 * np.power(gan_config.paint_points[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % predict_for_target.data.numpy().mean(),
                 fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -loss_discriminator.data.numpy(), fontdict={'size': 15})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=12);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()
