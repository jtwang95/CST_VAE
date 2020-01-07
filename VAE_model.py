from networks import *
import math
import numpy as np


# P
class Layer_Generator(nn.Module):
    def __init__(self,
                 batch_size=32,
                 image_channel=3,
                 canvas_size=64,
                 z_dim=128,
                 theta_parametrization="222"):
        super().__init__()
        self.canvas_size = canvas_size
        self.z_dim = z_dim
        self.image_channel = image_channel
        self.batch_size = batch_size
        self.theta_parametrization = theta_parametrization

        ## basic modules
        self.G_pose = Generator_Pose(
            image_channel=self.image_channel,
            canvas_size=self.canvas_size,
            theta_parametrization=self.theta_parametrization)
        self.G_style = Generator_Style(image_channel=self.image_channel,
                                       canvas_size=self.canvas_size)
        self.STNet = STNet()

    def forward(self, z_style, z_pose):
        # z_pose = torch.empty([self.batch_size, self.z_dim]).normal_()
        theta = self.G_pose(z_pose)
        # z_style = torch.empty([self.batch_size, self.z_dim]).normal_()
        C, alpha = self.G_style(z_style)
        return self.STNet(C, theta), self.STNet(
            alpha, theta), [C, theta, z_style, z_pose]


# Q
class Layer_Inference(nn.Module):
    def __init__(self,
                 batch_size=32,
                 image_channel=3,
                 canvas_size=64,
                 z_dim=128,
                 theta_parametrization="222"):
        super().__init__()
        self.canvas_size = canvas_size
        self.z_dim = z_dim
        self.image_channel = image_channel
        self.batch_size = batch_size

        ## basic modules
        self.E_pose = Encoder_Pose(image_channel=self.image_channel)
        self.E_style = Encoder_Style(image_channel=self.image_channel)
        self.STNet = STNet()
        self.InverseTheta = InverseTheta()

    def forward(self, X, G_pose):
        z_mu_pose, z_logvar_pose = self.E_pose(X)
        z_pose = reparameterization(z_mu_pose, z_logvar_pose)
        theta = G_pose(z_pose)
        theta_inv = self.InverseTheta(theta)
        X_canonical = self.STNet(X, theta_inv)
        z_mu_style, z_logvar_style = self.E_style(X_canonical)
        z_style = reparameterization(z_mu_style, z_logvar_style)
        return z_style, z_pose, [
            z_mu_style, z_logvar_style, z_mu_pose, z_logvar_pose
        ]


class CST_VAE(nn.Module):
    def __init__(self,
                 image_channel=3,
                 z_dim=128,
                 lr=3e-4,
                 num_slots=5,
                 canvas_size=64,
                 beta=1.0,
                 gamma=1.0,
                 theta_parametrization="222"):
        super().__init__()
        self.z_dim = z_dim
        self.num_slots = num_slots
        self.beta = beta
        self.gamma = gamma
        self.canvas_size = canvas_size
        self.image_channel = image_channel
        self.theta_parametrization = theta_parametrization

        self.LayerGen = Layer_Generator(
            image_channel=self.image_channel,
            canvas_size=self.canvas_size,
            theta_parametrization=self.theta_parametrization)
        self.LayerInf = Layer_Inference(image_channel=self.image_channel,
                                        canvas_size=self.canvas_size)
        self.STNet = STNet()
        self.Composer = Compose()

    def forward_path(self, img):
        self.loss_KLD = 0.0
        self.loss_LLK = 0.0
        imgs_recons = torch.zeros_like(img)
        D_k = img
        for k in range(self.num_slots):
            z_style_k, z_pose_k, [
                z_mu_style_k, z_logvar_style_k, z_mu_pose_k, z_logvar_pose_k
            ] = self.LayerInf(D_k, self.LayerGen.G_pose)
            L_k, alpha_k, _ = self.LayerGen(z_style_k, z_pose_k)

            ## kld = \int q*log(q/p) dx
            self.loss_KLD += -0.5 * (
                -2.0 - z_logvar_pose_k - z_logvar_style_k +
                z_mu_pose_k.pow(2) + z_mu_style_k.pow(2) +
                z_logvar_pose_k.exp() + z_logvar_style_k.exp()).sum()
            # self.loss_KLD += -0.5 * (-2.0 - z_logvar_pose_k -
            #                          z_logvar_style_k).sum()
            # self.loss_LLK += -1.0 * ((D_k - L_k).pow(2) /
            #                          (2.0 * 0.1 * 0.1) + math.log(0.1)).sum()
            imgs_recons = self.Composer(imgs_recons, L_k, alpha_k)
            ## update residual of observed image
            D_k = F.relu(D_k - L_k)
        self.loss_LLK += -1.0 * ((img - imgs_recons).pow(2) /
                                 (2.0 * 0.1 * 0.1) + math.log(0.1)).sum()
        return imgs_recons

    def loss_function(self, imgs):
        self.batch_size = imgs.shape[0]
        imgs_recons = self.forward_path(imgs)
        loss = -1.0 * (self.loss_KLD + self.loss_LLK) / self.batch_size
        return loss

    def generation(self, bs=1):
        X = torch.zeros(
            [bs, self.image_channel, self.canvas_size,
             self.canvas_size]).cuda()
        for k in range(self.num_slots):
            z_pose = torch.empty([bs, self.z_dim]).normal_().cuda()
            z_style = torch.empty([bs, self.z_dim]).normal_().cuda()
            x_k, alpha_k, _ = self.LayerGen(z_style, z_pose)
            X = self.Composer(X, x_k, alpha_k)
        return X.cpu().numpy()

    def visualize(self, img):
        if img.shape[1] == 1:
            output = img.repeat([1, 3, 1, 1]).cpu().numpy()
        else:
            output = img.cpu().numpy()

        imgs_recons = torch.zeros_like(img)
        D_k = img
        for k in range(self.num_slots):
            z_style_k, z_pose_k, [
                z_mu_style_k, z_logvar_style_k, z_mu_pose_k, z_logvar_pose_k
            ] = self.LayerInf(D_k, self.LayerGen.G_pose)
            L_k, alpha_k, [C, theta, _, _] = self.LayerGen(z_style_k, z_pose_k)

            # add box
            if C.shape[1] == 1:
                C = C.repeat([1, 3, 1, 1])
            C[:, 0, 0, :] = 1.0
            C[:, 0, 63, :] = 1.0
            C[:, 0, :, 0] = 1.0
            C[:, 0, :, 63] = 1.0

            L_k_box = self.STNet(C, theta)
            output = np.concatenate([output, L_k_box.cpu().numpy()], 0)
            imgs_recons = self.Composer(imgs_recons, L_k_box, alpha_k)
            ## update residual of observed image
            D_k = F.relu(D_k - L_k)
        output = np.concatenate([output, imgs_recons.cpu().numpy()], 0)

        return output


class CST_VAE_s1r1sh2(nn.Module):
    def __init__(self,
                 image_channel=3,
                 z_dim=128,
                 lr=3e-4,
                 num_slots=5,
                 canvas_size=64,
                 beta=1.0,
                 gamma=1.0):
        super().__init__()
        self.z_dim = z_dim
        self.num_slots = num_slots
        self.beta = beta
        self.gamma = gamma
        self.canvas_size = canvas_size
        self.image_channel = image_channel

        self.LayerGen = Layer_Generator(image_channel=self.image_channel,
                                        canvas_size=self.canvas_size)
        self.LayerInf = Layer_Inference(image_channel=self.image_channel,
                                        canvas_size=self.canvas_size)
        self.STNet = STNet()
        self.Composer = Compose()

    def forward_path(self, img):
        self.loss_KLD = 0.0
        self.loss_LLK = 0.0
        imgs_recons = torch.zeros_like(img)
        D_k = img
        for k in range(self.num_slots):
            z_style_k, z_pose_k, [
                z_mu_style_k, z_logvar_style_k, z_mu_pose_k, z_logvar_pose_k
            ] = self.LayerInf(D_k, self.LayerGen.G_pose)
            L_k, alpha_k, _ = self.LayerGen(z_style_k, z_pose_k)

            ## kld = \int q*log(q/p) dx
            self.loss_KLD += -0.5 * (
                -2.0 - z_logvar_pose_k - z_logvar_style_k +
                z_mu_pose_k.pow(2) + z_mu_style_k.pow(2) +
                z_logvar_pose_k.exp() + z_logvar_style_k.exp()).sum()
            # self.loss_KLD += -0.5 * (-2.0 - z_logvar_pose_k -
            #                          z_logvar_style_k).sum()
            # self.loss_LLK += -1.0 * ((D_k - L_k).pow(2) /
            #                          (2.0 * 0.1 * 0.1) + math.log(0.1)).sum()
            imgs_recons = self.Composer(imgs_recons, L_k, alpha_k)
            ## update residual of observed image
            D_k = F.relu(D_k - L_k)
        self.loss_LLK += -1.0 * ((img - imgs_recons).pow(2) /
                                 (2.0 * 0.1 * 0.1) + math.log(0.1)).sum()
        return imgs_recons

    def loss_function(self, imgs):
        self.batch_size = imgs.shape[0]
        imgs_recons = self.forward_path(imgs)
        loss = -1.0 * (self.loss_KLD + self.loss_LLK) / self.batch_size
        return loss

    def generation(self, bs=1):
        X = torch.zeros(
            [bs, self.image_channel, self.canvas_size,
             self.canvas_size]).cuda()
        for k in range(self.num_slots):
            z_pose = torch.empty([bs, self.z_dim]).normal_().cuda()
            z_style = torch.empty([bs, self.z_dim]).normal_().cuda()
            x_k, alpha_k, _ = self.LayerGen(z_style, z_pose)
            X = self.Composer(X, x_k, alpha_k)
        return X.cpu().numpy()

    def visualize(self, img):
        if img.shape[1] == 1:
            output = img.repeat([1, 3, 1, 1]).cpu().numpy()
        else:
            output = img.cpu().numpy()

        imgs_recons = torch.zeros_like(img)
        D_k = img
        for k in range(self.num_slots):
            z_style_k, z_pose_k, [
                z_mu_style_k, z_logvar_style_k, z_mu_pose_k, z_logvar_pose_k
            ] = self.LayerInf(D_k, self.LayerGen.G_pose)
            L_k, alpha_k, [C, theta, _, _] = self.LayerGen(z_style_k, z_pose_k)

            # add box
            if C.shape[1] == 1:
                C = C.repeat([1, 3, 1, 1])
            C[:, 0, 0, :] = 1.0
            C[:, 0, 63, :] = 1.0
            C[:, 0, :, 0] = 1.0
            C[:, 0, :, 63] = 1.0

            L_k_box = self.STNet(C, theta)
            output = np.concatenate([output, L_k_box.cpu().numpy()], 0)
            imgs_recons = self.Composer(imgs_recons, L_k_box, alpha_k)
            ## update residual of observed image
            D_k = F.relu(D_k - L_k)
        output = np.concatenate([output, imgs_recons.cpu().numpy()], 0)

        return output


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = CST_VAE(image_channel=1).to(device)

    img = torch.empty([2, 1, 64, 64]).normal_().to(device)
    X = model.forward_path(img)
    #plt.imshow(X.detach().permute([0, 2, 3, 1]).numpy()[0])
    #plt.savefig("test.png")
    print(X.shape)