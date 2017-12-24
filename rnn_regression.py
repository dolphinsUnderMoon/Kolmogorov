import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class Config:
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 2e-2
        self.time_step = 10
        self.input_size = 1
        self.training_max_iterations = 50
        self.show_period = 50


rnn_config = Config()

# anchors = np.linspace(0, np.pi*2, 100, dtype=np.float32)
# x_np = np.sin(anchors)
# y_np = np.cos(anchors)
# plt.plot(anchors, y_np, 'r-', label='target (cos)')
# plt.plot(anchors, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=rnn_config.input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.output = nn.Linear(32, 1)

    def forward(self, x, h_state):
        rnn_out, h_state = self.rnn(x, h_state)

        outs = []
        for time_step in range(rnn_out.size(1)):
            outs.append(self.output(rnn_out[:, time_step, :]))

        return torch.stack(outs, dim=1), h_state


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=rnn_config.learning_rate)
loss_function = nn.MSELoss()

plt.figure(1, figsize=(12, 5))
plt.ion()

h_state = None

for step in range(rnn_config.training_max_iterations):
    start, end = step * np.pi, (step+1) * np.pi

    steps = np.linspace(start, end, rnn_config.time_step, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data)

    loss = loss_function(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()
