from utils import set_seed_device, generate_harmonic_oscillator, plot_phase_portrait
from model import ForecastNet
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data as Data
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--nEpoch', default=100, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')

parser.add_argument('--seq_len', default=20, type=int, help='number of time-steps')
parser.add_argument('--in_features', default=2, type=int, help='input dimension')

parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size', default=64, type=int, help='dimensionality of hidden layer')


opt = parser.parse_args()
opt.device = set_seed_device(opt.seed)

# -------------- generate train and test set --------------
train_set, test_set = generate_harmonic_oscillator(delta_t=.33)
# plot_phase_portrait(test_set, 20, fname='test_set_portrait_delta_t_1')

train_dataset = Data.TensorDataset(train_set[:, :-1], train_set[:, 1:])  # input, labels
test_dataset = Data.TensorDataset(test_set[:, :-1], test_set[:, 1:])  # input, labels
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

# -------------- initialize model & optimizer --------------
model = ForecastNet(opt.in_features, opt.rnn_size).to(opt.device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# -------------- train & eval the model --------------
total_tr_losses, total_te_losses = [], []
for epoch in tqdm(range(opt.nEpoch)):
    losses_train, losses_test = [], []
    model.train()
    for x, target in train_loader:

        x, target = x.to(opt.device).float(), target.to(opt.device).float()
        optimizer.zero_grad()
        output = model(x)
        loss = model.loss(output, target)

        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())

    # validation iter
    model.eval()
    with torch.no_grad():
        for x, target in test_loader:
            x, target = x.to(opt.device).float(), target.to(opt.device).float()
            output = model(x)
            loss = model.loss(output, target)
            losses_test.append(loss.item())

    print('Train loss: {:.6f} | Test loss: {:.6f}'.format(np.mean(losses_train), np.mean(losses_test)))
    total_tr_losses.append(np.mean(losses_train))
    total_te_losses.append(np.mean(losses_test))

# -------------- plot loss during train --------------
plt.plot(range(len(total_tr_losses)), total_tr_losses, label='train')
plt.plot(range(len(total_te_losses)), total_te_losses, label='test')
plt.legend()
plt.yscale("log")
plt.title("Train and Test Loss During Training")
plt.savefig(f'./results/train_net.pdf', transparent=True, bbox_inches='tight', pad_inches=0,
            dpi=300)
plt.show()
#
#
# -------------- plot inference of trajectories --------------
n_trajectories = 10000


def inference(x_0, v_0=torch.zeros(1)):
    model.eval()
    x_0, v_0 = x_0.to(opt.device), v_0.to(opt.device)
    predicted = torch.Tensor([[x_0, v_0]]).to(opt.device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(opt.seq_len - 1):
            output = model(predicted)
            predicted = torch.cat((predicted, output[:, -1].unsqueeze(0)), dim=1)
    return predicted


data_inf = []
for i in range(n_trajectories):
    x_0 = torch.ones(1) * np.random.rand()
    infered_data = inference(x_0)
    data_inf.append(infered_data)

plot_phase_portrait(torch.cat(data_inf).detach().cpu(), n_trajectories)