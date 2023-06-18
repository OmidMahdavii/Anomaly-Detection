import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class SensorReadings(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        x, y = self.examples[index]
        return torch.Tensor(x), torch.Tensor(y)


def normalize(data, min, max):
    return (data - min) / (max - min)


def window_plot(window):
    fig, ax = plt.subplots()
    ax.plot(window[:, 0], color='b', label='Apparent power')
    ax.plot(window[:, 1], color='r', label='Current')
    ax.plot(window[:, 2], color='g', label='Frequency')
    ax.plot(window[:, 3], color='y', label='Phase angle')
    fig.set_size_inches(5, 4)
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Value')
    ax.legend()
    plt.draw()


def windowing(data, window_size, overlapping=True):
    # Stride is equal to the window size if the windows don't overlap
    stride = (window_size // 3) if overlapping else window_size
    output = [data[i:i+window_size, :] for i in range(0, data.shape[0]-window_size+1, stride)]
    if data.shape[0] % stride != 0:
        # Add a complete window from the end
        output.append(data[-window_size:, :])
    return np.array(output)


def build_splits(opt):
    normal_data = np.load('data/KukaNormal.npy')
    slow_data = np.load('data/KukaSlow.npy')[:, :-1] # The last column is anomaly label. It is not needed

    # Normalize data for each sensor reading in [0, 1]
    # Max and min are computed only for the training data
    train_max = np.max(normal_data, axis=0) + 1e-6 # add a small constant to avoid divide by 0 when max=min
    train_min = np.min(normal_data, axis=0)
    normal_data_normalized = normalize(normal_data, train_min, train_max)
    slow_data_normalized = normalize(slow_data, train_min, train_max)

    normal_data_window = windowing(normal_data_normalized, opt['window_size'], overlapping=True)
    # Test data windows don't have overlap
    slow_data_window = windowing(slow_data_normalized, opt['window_size'], overlapping=False)

    # 10% of the normal data used for validation and the rest used for training
    normal_split_index = normal_data_window.shape[0] // 10

    # 10% of the slow data used for validation and the rest used for test
    slow_split_index = slow_data_window.shape[0] // 10
    
    # window_plot(normal_data_window[10][:, 1:5])
    # window_plot(slow_data_window[274][:, 1:5])
    # plt.show()

    # Shuffle the data before splitting
    np.random.seed(42)
    np.random.shuffle(normal_data_window)
    np.random.shuffle(slow_data_window)

    train_examples = []
    val_examples = []
    test_examples = []

    # Normal data is labeled with 0
    for idx, example in enumerate(normal_data_window):
        if idx <= normal_split_index:
            val_examples.append((example, [0]))
        else:
            train_examples.append((example, [0]))

    # Abnormal data is labeled with 1
    for idx, example in enumerate(slow_data_window):
        if idx <= slow_split_index:
            val_examples.append((example, [1]))
        else:
            test_examples.append((example, [1]))

    # DataLoaders
    train_loader = DataLoader(SensorReadings(train_examples), batch_size=opt['batch_size'], shuffle=True)
    val_loader = DataLoader(SensorReadings(val_examples), batch_size=opt['batch_size'], shuffle=False)
    test_loader = DataLoader(SensorReadings(test_examples), batch_size=opt['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader