import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np

LEAKYRELU_SLOPE = 0.01
DROP_RATE = 0.2


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        m.bias.data.fill_(0.00)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        m.bias.data.fill_(0.00)


def reparameterization(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + eps * std


## input: x [bs,c,w,h] theta [bs,2,3]
## output: xs after spatial transormation
class STNet(nn.Module):
    # Spatial transformer operation
    def forward(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        xs = F.grid_sample(x, grid)
        return xs


## input: theta [bs,2,3]
class InverseTheta(nn.Module):
    def forward(self, theta):
        try:
            aug = torch.empty([theta.shape[0], 1, 3]).zero_()
            aug[:, :, 2] = 1.0
            theta_aug = torch.cat((theta, aug), axis=1)
        except RuntimeError:
            aug = torch.empty([theta.shape[0], 1, 3]).zero_().cuda()
            aug[:, :, 2] = 1.0
            theta_aug = torch.cat((theta, aug), axis=1)
        return theta_aug.inverse()[:, :2, :]


## input: x,_x,alpha
class Compose(nn.Module):
    def forward(self, x, _x, alpha):
        #print(x.device)
        return x * (1.0 - alpha) + _x * alpha


## input: z_theta [bs,z_dim]
## output: theta [bs,2,3]
class Generator_Pose(nn.Module):
    def __init__(self,
                 image_channel=3,
                 canvas_size=64,
                 z_dim=128,
                 theta_parametrization="222"):
        super().__init__()
        self.canvas_size = canvas_size
        self.z_dim = z_dim
        self.image_channel = image_channel
        self.theta_parametrization = theta_parametrization

        # Regressor for the 3 * 2 affine matrix
        if self.theta_parametrization == "222":
            self.thetalization = nn.Sequential(nn.Linear(self.z_dim, 128),
                                               nn.BatchNorm1d(128),
                                               nn.ReLU(True),
                                               nn.Linear(128, 256),
                                               nn.BatchNorm1d(256),
                                               nn.ReLU(True),
                                               nn.Linear(256, 3 * 2))

            # Initialize the weights/bias with identity transformation
            self.thetalization[-1].weight.data.mul_(0.0001)
            self.thetalization[-1].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.theta_parametrization == "112":
            self.thetalization = nn.Sequential(nn.Linear(self.z_dim, 128),
                                               nn.BatchNorm1d(128),
                                               nn.ReLU(True),
                                               nn.Linear(128, 256),
                                               nn.BatchNorm1d(256),
                                               nn.ReLU(True))

            self.scalelization = nn.Sequential(nn.Linear(256, 1))
            self.scalelization[-1].weight.data.mul_(0.001)
            self.scalelization[-1].bias.data.uniform_(1.0, 2.0)

            self.anglelization = nn.Sequential(nn.Linear(256, 1))
            self.anglelization[-1].weight.data.mul_(0.001)
            self.anglelization[-1].bias.data.uniform_(-0.1, 0.1)

            self.shiftlization = nn.Sequential(nn.Linear(256, 2))
            self.shiftlization[-1].weight.data.mul_(0.001)
            self.shiftlization[-1].bias.data.uniform_(-0.1, 0.1)
            # Initialize the weights/bias with identity transformation
            # self.thetalization[-1].weight.data.mul_(0.001)
            # # self.thetalization[-1].bias.data.copy_(
            # #     torch.tensor([2.0, 0, 0, 0], dtype=torch.float))
            # self.thetalization[-1].bias.data.uniform_(-1.0, 1.0)

    def forward(self, z):
        if self.theta_parametrization == "222":
            #print("222")
            theta = self.thetalization(z).view(-1, 2, 3)
            theta[:, :, 2] = nn.Tanh()(theta[:, :, 2])
        elif self.theta_parametrization == "112":
            #print("112")
            theta = torch.empty([z.shape[0], 2, 3]).zero_().cuda()
            output = self.thetalization(z)
            angle = nn.Tanh()(self.anglelization(output)) * math.pi
            scale = nn.Softplus()(self.scalelization(output))
            # print(scale.shape)
            # print(torch.cos(angle).shape)
            theta[:, 0, 0] = (scale * torch.cos(angle))[:, 0]
            theta[:, 0, 1] = (scale * torch.sin(angle))[:, 0]
            theta[:, 1, 0] = -1.0 * (scale * torch.sin(angle))[:, 0]
            theta[:, 1, 1] = (scale * torch.cos(angle))[:, 0]
            theta[:, :, 2] = nn.Tanh()(self.shiftlization(output))
        return theta


## input: z_post [bs,z_dim]
## output: rgb+alpha [bs,3+1,w,h]
class Generator_Style(nn.Module):
    def __init__(self, image_channel=3, canvas_size=64, z_dim=128):
        super().__init__()
        self.canvas_size = canvas_size
        self.z_dim = z_dim
        self.image_channel = image_channel
        self.netG = nn.Sequential(nn.Conv2d(self.z_dim, 2048, 1, 1),
                                  nn.BatchNorm2d(2048),
                                  nn.LeakyReLU(LEAKYRELU_SLOPE),
                                  nn.Conv2d(2048, 256, 1, 1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(LEAKYRELU_SLOPE),
                                  nn.ConvTranspose2d(256, 256, 4, 1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(LEAKYRELU_SLOPE),
                                  nn.ConvTranspose2d(256, 128, 4, 2),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(LEAKYRELU_SLOPE),
                                  nn.ConvTranspose2d(128, 128, 4, 1),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(LEAKYRELU_SLOPE),
                                  nn.ConvTranspose2d(128, 64, 4, 2),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(LEAKYRELU_SLOPE),
                                  nn.ConvTranspose2d(64, 64, 4, 1),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(LEAKYRELU_SLOPE),
                                  nn.ConvTranspose2d(64, 64, 4, 2),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(LEAKYRELU_SLOPE),
                                  nn.Conv2d(64, self.image_channel + 1, 1, 1),
                                  nn.Sigmoid())

    def forward(self, z):
        z = z.view([-1, self.z_dim, 1, 1])
        output = self.netG(z)
        return output[:,
                      0:self.image_channel, :, :], output[:,
                                                          self.image_channel:
                                                          (self.image_channel +
                                                           1), :, :]


class Encoder_Pose(nn.Module):
    def __init__(self, image_channel=3, canvas_size=64, z_dim=128):
        super().__init__()
        self.canvas_size = canvas_size
        self.z_dim = z_dim
        self.image_channel = image_channel

        self.localization = nn.Sequential(
            nn.Conv2d(self.image_channel, 64, 4, 2), nn.BatchNorm2d(64),
            nn.LeakyReLU(LEAKYRELU_SLOPE), nn.Conv2d(64, 64, 4, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(LEAKYRELU_SLOPE),
            nn.Conv2d(64, 128, 4, 2), nn.BatchNorm2d(128),
            nn.LeakyReLU(LEAKYRELU_SLOPE), nn.Conv2d(128, 128, 4, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(LEAKYRELU_SLOPE),
            nn.Conv2d(128, 256, 4, 2), nn.BatchNorm2d(256),
            nn.LeakyReLU(LEAKYRELU_SLOPE), nn.Conv2d(256, 128, 4, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(LEAKYRELU_SLOPE), Flatten())

        self.z_poselization = nn.Sequential(nn.Linear(128, 256),
                                            nn.BatchNorm1d(256), nn.ReLU(True),
                                            nn.Linear(256, self.z_dim * 2))

    def forward(self, X):
        output = self.localization(X)
        output = self.z_poselization(output)
        return output


class Encoder_Style(nn.Module):
    def __init__(self, image_channel=3, canvas_size=64, z_dim=128):
        super().__init__()
        self.canvas_size = canvas_size
        self.z_dim = z_dim
        self.image_channel = image_channel

        self.z_stylization = nn.Sequential(
            nn.Conv2d(self.image_channel, 64, 4, 2), nn.BatchNorm2d(64),
            nn.LeakyReLU(LEAKYRELU_SLOPE), nn.Conv2d(64, 64, 4, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(LEAKYRELU_SLOPE),
            nn.Conv2d(64, 128, 4, 2), nn.BatchNorm2d(128),
            nn.LeakyReLU(LEAKYRELU_SLOPE), nn.Conv2d(128, 128, 4, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(LEAKYRELU_SLOPE),
            nn.Conv2d(128, 256, 4, 2), nn.BatchNorm2d(256),
            nn.LeakyReLU(LEAKYRELU_SLOPE), nn.Conv2d(256, 1024, 4, 1),
            nn.BatchNorm2d(1024), nn.LeakyReLU(LEAKYRELU_SLOPE), Flatten(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Linear(256, self.z_dim * 2))

    def forward(self, X):
        output = self.z_stylization(X)
        return output[:, self.z_dim:], output[:, :self.z_dim]

class LSTM_LatentPose(nn.Module):
    def __init__(self,input_size=256,output_size=256,hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size=self.input_size,hidden_size=self.hidden_size)
        self.rnn_linear = nn.Sequential(nn.Linear(self.hidden_size,self.output_size))

    def forward(self,z_x,state):
        state = self.rnn(z_x,state)
        output = self.rnn_linear(state[0])
        return output,state
        

if __name__ == "__main__":
    # G_pose = Generator_Pose()
    # G_pose.apply(weights_init)
    # G_pose.eval()
    # G_style = Generator_Style()
    # G_style.apply(weights_init)
    # G_style.eval()
    # STN = STNet()

    # fig, ax = plt.subplots(1, 3)

    # x = torch.empty([1, 3, 64, 64]).zero_()
    # x[0, 0, 28:36, 28:36] = 1.0
    # ax[0].imshow(x.detach().permute([0, 2, 3, 1]).numpy()[0])

    # z = torch.empty([1, 128]).normal_()
    # theta = G_pose(z)
    # print(theta)
    # scale = 1.0
    # angle = 30 * math.pi / 180
    # theta[0, 0, 0] = scale * math.cos(angle)
    # theta[0, 1, 1] = scale * math.cos(angle)
    # theta[0, 0, 1] = scale * math.sin(angle)
    # theta[0, 1, 0] = -math.sin(angle) * scale
    # theta[0, 0, 2] = -0.5
    # theta[0, 1, 2] = -0.5
    # #x = G_style(z)
    # xs = STN(x, theta)
    # ax[1].imshow(xs.detach().permute([0, 2, 3, 1]).numpy()[0])

    # theta_back = InverseTheta()(theta)
    # xsp = STN(xs, theta_back)
    # ax[2].imshow(xsp.detach().permute([0, 2, 3, 1]).numpy()[0])
    # plt.savefig("./test.png")
    #print(theta)
    #print(theta_back)

    # E_pose = Encoder_Pose()
    # E_pose.apply(weights_init)
    # E_pose.eval()
    # E_style = Encoder_Style()
    # E_style.apply(weights_init)
    # E_style.eval()

    # X = torch.empty([1, 3, 64, 64]).normal_()
    # z_pose = E_pose(X)
    # z_style = E_style(X)
    # print(z_pose[0])
    # print(z_style[0])
    x = torch.empty([1, 3, 64, 64]).zero_().cuda()
    x[0, 0, 28:36, 28:36] = 1.0
    Gpose = Generator_Pose(theta_parametrization="112")
    Gpose.eval()
    z = torch.empty([1, 128]).normal_()
    theta = Gpose(z)
    STN = STNet()
    xs = STN(x, theta)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.detach().cpu().permute([0, 2, 3, 1]).numpy()[0])
    ax[1].imshow(xs.detach().cpu().permute([0, 2, 3, 1]).numpy()[0])
    plt.savefig("test.png")
    print(theta)
