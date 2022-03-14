import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEncoder(nn.Module):

    def __init__(self, input_size, z_size):
        super(LinearEncoder, self).__init__()

        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 128)

        self.z_mean = nn.Linear(128, z_size)
        self.z_scale = nn.Linear(128, z_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)

        z_mu = self.z_mean(x)
        z_logvar = self.z_scale(x)

        return z_mu, z_logvar


class LinearDecoder(nn.Module):

    def __init__(self, input_size, z_size):
        super(LinearDecoder, self).__init__()

        self.fc1 = nn.Linear(z_size, 128)
        self.fc2 = nn.Linear(128, 400)
        self.fc3 = nn.Linear(400, input_size)

    def forward(self, x):

        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        output = F.elu(x)

        return output


class ImageEncoder(nn.Module):

    def __init__(self, image_height, z_size):
        super(ImageEncoder, self).__init__()

        self.image_height = image_height
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn_conv2 = nn.BatchNorm2d(32)
        fc1_size = int(32 * (self.image_height / 2**2)**2)
        self.fc1 = nn.Linear(fc1_size, 400)
        self.bn_fc1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.z_mu = nn.Linear(128, z_size)
        self.z_scale = nn.Linear(128, z_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn_conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn_conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn_fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn_fc2(x)
        z_loc = self.z_mu(x)
        z_logvar = self.z_scale(x)

        return z_loc, z_logvar


class ImageDecoder(nn.Module):

    def __init__(self, image_height, z_size):
        super(ImageDecoder, self).__init__()
        self.image_height = image_height
        self.fc1 = nn.Linear(z_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 400)
        self.bn_fc2 = nn.BatchNorm1d(400)
        fc2_size = int(32 * (self.image_height / 2**2)**2)
        self.fc3 = nn.Linear(400, fc2_size)
        self.bn_fc3 = nn.BatchNorm1d(fc2_size)

        self.conv1 = nn.ConvTranspose2d(32, 64, 3, 2, 1, 1)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 3, 3, 2, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn_fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn_fc2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn_fc3(x)

        x = x.view(x.size()[0], 32, int(self.image_height / 2**2), int(self.image_height / 2**2))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn_conv1(x)
        x = self.conv2(x)
        output = torch.sigmoid(x)

        return output
