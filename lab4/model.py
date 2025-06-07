import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(num_maps_in, num_maps_out, k, padding=k // 2, bias=bias))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = _BNReluConv(input_channels, emb_size)
        self.conv2 = _BNReluConv(input_channels, emb_size)
        self.conv3 = _BNReluConv(input_channels, emb_size)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def get_features(self, img):
        x = self.conv1(img)
        x = self.maxpool(x)
        x = self.conv2(img)
        x = self.maxpool(x)
        x = self.conv3(img)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def loss(self, anchor, positive, negative, margin=1):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        positive_distance = torch.norm(a_x - p_x, p=2, dim=1)
        negative_distance = torch.norm(a_x - n_x, p=2, dim=1)
        
        loss = F.relu(margin + positive_distance - negative_distance).mean()
        return loss



class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        
        feats = img.view(img.size(0), -1)
        return feats
