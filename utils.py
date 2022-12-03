import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('cuda available')
    else:
        device = torch.device("cpu")
    return device


def compute_velocity(x_next, x_prev, delta_t):
    return (x_next - x_prev) / delta_t


def verlet_method(x_0, delta_t=0.1, seq_len=20):
    v_0 = 0
    x_1 = x_0 + v_0 * delta_t - 0.5 * x_0 * (delta_t ** 2)
    v_1 = compute_velocity(x_1, x_0, delta_t)
    x = [x_0, x_1]
    velocity = [v_0, v_1]
    for _ in range(seq_len - 2):
        x_t_p = x[-2]  # x_(t-1)
        x_t = x[-1]  # x_t
        x_next = 2 * x_t - x_t_p - x_t * (delta_t ** 2)  # x_(t+1)
        x.append(x_next)
        velocity.append(compute_velocity(x_next, x_t, delta_t))
    return np.array([(xx, vv) for xx, vv in zip(x, velocity)])


def generate_harmonic_oscillator(n=500, delta_t=.1):
    ho_dataset = []
    np.random.seed(1)
    for _ in range(n):
        x_0 = np.random.rand()
        ho_dataset.append(torch.from_numpy(verlet_method(x_0, delta_t)))
    ho_dataset = torch.stack(ho_dataset)
    train = ho_dataset[:int(0.8 * n)]
    test = ho_dataset[int(0.8 * n):]
    return train, test


def plot_phase_portrait(data, n_trajectories):
    for i in range(n_trajectories):
        plt.plot(data[i, :, 0], data[i, :, 1])
        # U = x(t) - x(t-1) for any t.
        # V = v(t) - v(t-1) for any t.
        U = torch.cat((data[i, 1:, 0] - data[i, :-1, 0], data[i, 0, 0] - data[i, -1, 0][None]))
        V = torch.cat((data[i, 1:, 1] - data[i, :-1, 1], data[i, 0, 1] - data[i, -1, 1][None]))
        plt.quiver(data[i, :, 0], data[i, :, 1], U, V)
    plt.savefig(f'./results/phase_portrait_{n_trajectories}.pdf', transparent=True, bbox_inches='tight', pad_inches=0,
                dpi=300)
    plt.show()
