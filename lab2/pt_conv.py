import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
import skimage as ski
import skimage.io
import os
from pathlib import Path

SAVE_DIR = Path(__file__).parent / 'conv/filters'


def find_highest_experiment(directory: str):
    """
    Find the highest numbered experiment in a directory containing subdirectories named exp1, exp2, etc.
    """
    dir_path = Path(directory)
    
    max_exp = 0
    for subdir in dir_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith("exp"):
            try:
                exp_num = int(subdir.name[3:])
                max_exp = max(max_exp, exp_num)
            except ValueError:
                continue
    
    return max_exp


def draw_conv_filters(epoch, step, layer, name, save_dir, final=False):
  C = layer.weight.shape[1]
  w = layer.weight.detach().numpy()
  num_filters = w.shape[0]
  assert w.shape[2] == w.shape[3]
  k = int(np.sqrt(w.shape[2] * w.shape[3] / C))
  w = w.reshape(num_filters, C, k, k)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  #for i in range(C):
  for i in range(1):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    img = (img*255).astype(np.uint8)
    if final:
       filename = filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % ("final", epoch, step, i)
    else:
        filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % (name, epoch, step, i)
    ski.io.imsave(os.path.join(save_dir, filename), img)
 
class CovolutionalModel(torch.nn.Module):
  def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
    super(CovolutionalModel, self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    # ostatak konvolucijskih slojeva i slojeva sažimanja
    self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.ReLU1 = torch.nn.ReLU()
    self.conv2 = torch.nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.ReLU2 = torch.nn.ReLU()
    # potpuno povezani slojevi
    self.fc1 = torch.nn.Linear(7*7*conv2_width, fc1_width, bias=True)
    self.fc_logits = torch.nn.Linear(fc1_width, class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)
      elif isinstance(m, torch.nn.Linear) and m is not self.fc_logits:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.maxpool1(h)
    h = self.ReLU1(h)
    h = self.conv2(h)
    h = self.maxpool2(h)
    h = self.ReLU2(h)

    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    return  torch.softmax(logits, dim=1)


def train(model, train_dataloader, val_dataloader, param_niter=10000, param_delta=0.1, param_lambda=1e-2):
    optimizer = torch.optim.SGD(model.parameters(), lr = param_delta, weight_decay=param_lambda)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    if not os.path.exists("runs"):
        os.makedirs("runs")
    new_exp = find_highest_experiment("runs") + 1
    writer = SummaryWriter(f"runs/exp{new_exp}")
    
    for epoch in range(1, param_niter+1):
        model.train()
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch",
                  desc=f'Training (epoch={epoch}/{param_niter})') as t:   
            for i, (X, Y_) in t:
               Y = model(X)
               loss = F.cross_entropy(Y, Y_)
               loss.backward()
               optimizer.step()
               optimizer.zero_grad()
               writer.add_scalar("train_loss", loss.item(), epoch*len(train_dataloader) + i)
               t.set_postfix(loss=loss.item())
               if i % 1000 == 0:
                  draw_conv_filters(epoch, i*len(X), model.conv1, "conv1", SAVE_DIR)
        scheduler.step()
        model.eval()
        with torch.no_grad():
           for X, Y_ in val_dataloader:
              Y = model(X)
              loss = F.cross_entropy(Y, Y_)
              writer.add_scalar('val_loss', loss.item(), epoch * len(val_dataloader) + i)
    draw_conv_filters(epoch, i*len(X), model.conv1, "conv1", SAVE_DIR, True)
    writer.close()




if __name__=="__main__":
    dataset_root = '/tmp/mnist'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    n = len(mnist_train)
    mnist_list = random_split(mnist_train, [int(n * 0.8), n - int(n * 0.8)])
    mnist_train = mnist_list[0]
    mnist_val = mnist_list[1]

    indices_0 = mnist_list[0].indices
    indices_1 = mnist_list[1].indices


    x_train, y_train = mnist_train.dataset.data[indices_0], mnist_train.dataset.targets[indices_0]
    x_val, y_val = mnist_val.dataset.data[indices_1], mnist_val.dataset.targets[indices_1]
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_val, x_test = x_train.float().div_(255.0), x_val.float().div_(255.0), x_test.float().div_(255.0)

    train_data = TensorDataset(x_train.view(-1, 1, 28, 28), y_train)
    val_data = TensorDataset(x_val.view(-1, 1, 28, 28), y_val)
    train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=50, shuffle=False)

    model = CovolutionalModel(1, 16, 32, 512, 10)
    
    train(model, train_dataloader, val_dataloader, param_niter=8)


