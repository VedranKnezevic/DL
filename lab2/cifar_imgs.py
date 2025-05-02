import pt_cifar
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np


DATA_DIR = '/tmp/cifar10/cifar-10-batches-py/'
SAVE_DIR = Path(__file__).parent / 'cifar'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10


def show_image(image, title):
    plt.imshow(image.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__=="__main__":
    subset = pt_cifar.unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)


    model = pt_cifar.CifarNet()
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model_15epochs.pth')))
    model.eval()

    misclassified_images = []


    for i in range(len(test_x)):
        image = test_x[i]
        label = test_y
        output = model(torch.tensor(image))
        exit()
        loss = model.get_loss(output, label)

        _, predicted = torch.max(output, 1)
        if predicted != label:
            misclassified_images.append((image, label, predicted, loss.item(), output))

    misclassified_images.sort(key=lambda x: x[3], reverse=True)

    for i in range(20):
        image, true_label, predicted_label, loss, output = misclassified_images[i]
        true_class = ds_test.classes[true_label]
        predicted_classes = [ds_test.classes[predicted_label.item()] for predicted_label in torch.topk(output, 3).indices.squeeze()]

        show_image(image.cpu().squeeze(), f'True Class: {true_class}\nPredicted Classes: {predicted_classes}\nLoss: {loss}')



