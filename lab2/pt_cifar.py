import torch
from torch import nn
import torch.utils
from torch.utils.data import random_split, DataLoader, TensorDataset
import torchvision
import os
import numpy as np
import pickle
from pathlib import Path
import skimage as ski
import skimage.io
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


SAVE_DIR = Path(__file__).parent / 'cifar'
DATA_DIR = '/tmp/cifar10/cifar-10-batches-py/'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10
target_dict = {0: "airplane",
               1: "automobile",
               2: "bird",
               3: "cat",
               4: "deer",
               5: "dog",
               6: "frog",
               7: "horse",
               8: "ship",
               9: "truck"}



class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2)
        self.flat = nn.Flatten()
        
        self.fc1 = nn.Linear(32 * 7 * 7, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 10, bias=True)
      

    def forward(self, x):
        # print(x.shape)
        o = self.conv1(x)
        o = self.relu(o)
        o = self.maxpool(o)
        # print(o.shape)

        o = self.conv2(o)
        o = self.relu(o)
        o = self.maxpool(o)
        # print(o.shape)

        o = o.view(o.size(0), -1)
        # print(o.shape)
        # exit()
        o = self.fc1(o)
        o = self.relu(o)
        o = self.fc2(o)
        o = self.relu(o)
        o = self.fc3(o)

        return o
    

    def get_loss(self, y, target):
        y = torch.softmax(y, dim=1)
        return nn.functional.cross_entropy(y, target)


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = w[:,:,:,i]
    img = (img*255).astype(np.uint8)
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)



def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y


def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


def evaluate(model, data, loss_fn=nn.functional.cross_entropy):
    model.eval()

    with torch.no_grad():
        X, Y_ = data[:][0], data[:][1]
        Y = model(X)
        loss = loss_fn(Y, Y_)
        Y = torch.argmax(Y, dim=1)
        conf_matrix = np.zeros((num_classes, num_classes))
        acc = (Y == Y_).float().mean().item()
        for i in range(num_classes):
           for j in range(num_classes):
              conf_matrix[i,j] = np.logical_and((Y_==i).detach().numpy(), (Y==j).detach().numpy()).sum()
        
        precisions = conf_matrix.mean(0)
        recalls = conf_matrix.mean(1)

    return loss.item(), acc, conf_matrix, precisions, recalls
    

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


def train(model, train_data, val_data, num_epochs=15, lr=1e-4, param_lambda=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)
    train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True)

    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    for epoch in range(1, num_epochs+1):
        model.train()
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch",
                  desc=f'Training (epoch={epoch}/{num_epochs})') as t:
            for batch, (X, Y_) in t:
                Y = model(X)
                loss = model.get_loss(Y, Y_)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # if batch%100 == 0:    
                    # print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, batch, len(train_dataloader), loss))

                if batch%200 == 0:
                    draw_conv_filters(epoch, batch, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)


        train_loss, train_acc, _, _, _ = evaluate(model, train_data[:10000])
        val_loss, val_acc, _, _, _ = evaluate(model, val_data)

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [val_acc]
        plot_data['lr'] += [lr_scheduler.get_last_lr()]
        lr_scheduler.step()

    plot_training_progress(SAVE_DIR, plot_data)


if __name__=="__main__":

    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))
    test_x_numpy = test_x.copy()

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.transpose(0, 3, 1, 2)
    valid_x = valid_x.transpose(0, 3, 1, 2)
    test_x = test_x.transpose(0, 3, 1, 2)


    train_data = TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    val_data = TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))

    model = CifarNet()

    # train(model, train_data, val_data)

    # torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model_15epochs.pth'))

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model_15epochs.pth')))
    model.eval()

    image_results = []
    for i, (X, Y_) in enumerate(DataLoader(test_data)):
        out = model(X)

        top_three = [target_dict[x.item()] for x in torch.sort(out, descending=True)[1].squeeze()][:3]
        prediction = top_three[0]
        loss = nn.functional.cross_entropy(out, Y_)
        image = test_x_numpy[i].astype(np.int)
        image_results.append((image, loss.item(), target_dict[Y_.item()], top_three))
    

    image_results = sorted(image_results, key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(4, 5)

    ax = ax.ravel()
    plt.tight_layout()
    for i in range(20):
        ax[i].set_title(f"ground truth: {image_results[i][2]}\ntop three: {image_results[i][3]}\nloss: {image_results[i][1]}",
                        fontsize=10)
        ax[i].axis('off')
        ax[i].imshow(image_results[i][0])

    plt.show()

