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
        z_mu_logvar_pose = self.E_pose(X)
        z_mu_pose = z_mu_logvar_pose[:, :self.z_dim]
        z_logvar_pose = z_mu_logvar_pose[:, self.z_dim:]
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

        self.beta = 1.0

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


class CST_VAE_lstm(nn.Module):
    def __init__(self,
                 image_channel=3,
                 z_dim=128,
                 lr=3e-4,
                 num_slots=5,
                 canvas_size=64,
                 beta=1.0,
                 gamma=1.0,
                 theta_parametrization="222",
                 geco=False):
        super().__init__()
        self.z_dim = z_dim
        self.num_slots = num_slots
        self.beta = beta
        self.gamma = gamma
        self.canvas_size = canvas_size
        self.image_channel = image_channel
        self.theta_parametrization = theta_parametrization
        ## new parameters
        self.lstm_hidden = 256

        ## params for geco calculation
        self.geco = geco
        self.beta = 1.0  ## for geco calculation
        self.geco_lr = 1e-5 * 64**2 / self.canvas_size**2
        self.geco_speedup = None
        self.geco_alpha = 0.99
        self.goal_pixel = -0.35  ## goal per pixel
        self.c_ema = None  ## store for geco calculation

        # basic modules
        self.E_pose = Encoder_Pose(image_channel=self.image_channel)
        self.E_style = Encoder_Style(image_channel=self.image_channel)
        self.G_pose = Generator_Pose(
            image_channel=self.image_channel,
            canvas_size=self.canvas_size,
            theta_parametrization=self.theta_parametrization)
        self.G_style = Generator_Style(image_channel=self.image_channel,
                                       canvas_size=self.canvas_size)
        self.InverseTheta = InverseTheta()
        self.STNet = STNet()
        self.Composer = Compose()

        ## LSTM for modelling z_pose dependency
        ## X -> h[256]
        self.LSTM_qzPose = LSTM_LatentPose(input_size=256 + self.z_dim,
                                           output_size=2 * self.z_dim,
                                           hidden_size=self.lstm_hidden)
        self.LSTM_pzPose = LSTM_LatentPose(input_size=self.z_dim,
                                           output_size=2 * self.z_dim,
                                           hidden_size=self.lstm_hidden)
        self.Composer = Compose()

    def forward_path(self, img):
        D_k = img
        img_recons = torch.zeros_like(img)
        z_pose_pre = torch.zeros([D_k.shape[0], self.z_dim])

        ## store z_pose, z_style, z_mu_pose, z_logvar_pose, z_mu_style,z_logvar_style
        z_pose = []
        z_style = []
        z_mu_pose = []
        z_logvar_pose = []
        z_mu_style = []
        z_logvar_style = []

        ## latent state for lstm
        state = None

        for k in range(self.num_slots):
            ## inference z_pose from img. q(z_i|z_<i,X)
            if k == 0:
                h = self.E_pose(D_k)
                z_mu_pose_k = h[:, :self.z_dim]
                z_logvar_pose_k = h[:, self.z_dim:]
                z_pose_k = reparameterization(z_mu_pose_k, z_logvar_pose_k)
            else:
                h = self.E_pose(D_k)
                z_mu_logvar_pose_k, state = self.LSTM_qzPose(
                    torch.cat([h, z_pose_pre], dim=1), state)
                z_mu_pose_k = z_mu_logvar_pose_k[:, :self.z_dim]
                z_logvar_pose_k = z_mu_logvar_pose_k[:, self.z_dim:]
                z_pose_k = reparameterization(z_mu_pose_k, z_logvar_pose_k)

            ## store last z_pose for lstm
            z_pose_pre = z_pose_k

            ## inference z_style from img and z_pose. q(z_style_i|z_pose_i,X)
            theta_k = self.G_pose(z_pose_k)
            theta_k_inv = self.InverseTheta(theta_k)
            D_k_canonical = self.STNet(D_k, theta_k_inv)
            z_mu_style_k, z_logvar_style_k = self.E_style(D_k_canonical)
            z_style_k = reparameterization(z_mu_style_k, z_logvar_style_k)

            ## reconstruction
            C_k, alpha_k = self.G_style(z_style_k)
            L_k, alpha_k = self.STNet(C_k,
                                      theta_k), self.STNet(alpha_k, theta_k)
            img_recons = self.Composer(img_recons, L_k, alpha_k)

            ## store z for loss calculation
            z_pose.append(z_pose_k)
            z_style.append(z_style_k)
            z_mu_pose.append(z_mu_pose_k)
            z_logvar_pose.append(z_logvar_pose_k)
            z_mu_style.append(z_mu_style_k)
            z_logvar_style.append(z_logvar_style_k)

        return img_recons, z_pose, z_style, z_mu_pose, z_logvar_pose, z_mu_style, z_logvar_style

    def loss_function(self, imgs):
        self.batch_size = imgs.shape[0]
        imgs_recons, z_pose, z_style, z_mu_pose, z_logvar_pose, z_mu_style, z_logvar_style = self.forward_path(
            imgs)
        self.loss_KLD = 0.0
        self.loss_LLK = 0.0
        ## check dimension
        # if False:
        #     ## [K,bs,z_dim]
        #     print("imgs_recons:", len(imgs_recons), imgs_recons[0].shape)
        #     print("z_pose:", len(z_pose), z_pose[0].shape)
        #     print("z_style", len(z_style), z_style[0].shape)
        #     print("z_mu_pose", len(z_mu_pose), z_mu_pose[0].shape)
        #     print("z_logvar_pose", len(z_logvar_pose), z_logvar_pose[0].shape)
        #     print("z_mu_style", len(z_mu_style), z_mu_style[0].shape)
        #     print("z_logvar_style", len(z_logvar_style),
        #           z_logvar_style[0].shape)

        ## KLD
        ## KLD Eq[logq]
        for k in range(self.num_slots):
            #Eq(logq)
            self.loss_KLD -= -0.5 * (z_logvar_style[k] +
                                     (z_style[k] - z_mu_style[k]).pow(2) /
                                     z_logvar_style[k].exp()).sum()
            self.loss_KLD -= -0.5 * (z_logvar_pose[k] +
                                     (z_pose[k] - z_mu_pose[k]).pow(2) /
                                     z_logvar_pose[k].exp()).sum()
            ## KLD Eq(logp)
            ## KLD for Eq(logp(z_styple))
            self.loss_KLD += -0.5 * (z_style[k].pow(2)).sum()
        ## state for lstm
        state = None
        z_mu_pose_prior = [torch.zeros_like(z_pose[0])]
        z_logvar_pose_prior = [torch.zeros_like(z_pose[0])]
        for k in range(1, self.num_slots):
            z_pose_prior, state = self.LSTM_pzPose(z_pose[k - 1], state)
            z_mu_pose_prior.append(z_pose_prior[:, :self.z_dim])
            z_logvar_pose_prior.append(z_pose_prior[:, self.z_dim:])
        ## Eq(logp(z_pose))
        for k in range(len(z_mu_pose_prior)):
            self.loss_KLD += -0.5 * (z_logvar_pose_prior[k] +
                                     (z_pose[k] - z_mu_pose_prior[k]).pow(2) /
                                     z_logvar_pose_prior[k].exp()).sum()

        ## LLK
        self.loss_LLK = -1.0 * ((imgs - imgs_recons).pow(2) /
                                (2.0 * 0.1 * 0.1)).sum()

        ## batch mean
        self.loss_KLD = self.loss_KLD / self.batch_size
        self.loss_LLK = self.loss_LLK / self.batch_size

        if self.geco:
            ## TODO implement beta update algorithm
            ## loss = kld + beta(llk - tau)
            ## Initialize parameters
            #+++++++++++++++++++++++++++#
            num_pixels = imgs.shape[1] * imgs.shape[2] * imgs.shape[3]
            goal = self.goal_pixel * num_pixels
            #+++++++++++++++++++++++++++#
            loss_llk_new = self.loss_LLK.detach()
            self.c_ema = self.get_ema(loss_llk_new, self.c_ema,
                                      self.geco_alpha)

            self.beta = self.geco_beta_update(beta=self.beta,
                                              llk_ema=self.c_ema,
                                              goal=goal,
                                              step_size=self.geco_lr,
                                              speedup=self.geco_speedup)

            loss = -1.0 * self.beta * (self.loss_KLD) + (-1.0) * self.loss_LLK

        else:
            loss = -1.0 * (self.loss_KLD + self.loss_LLK)
        return loss

    def generation(self, bs=1):
        X = torch.zeros(
            [bs, self.image_channel, self.canvas_size,
             self.canvas_size]).cuda()
        ## state for lstm
        state = None
        for k in range(self.num_slots):
            if k == 0:
                z_pose_k = torch.empty([bs, self.z_dim]).normal_().cuda()
            else:
                z_mu_logvar_pose_k, state = self.LSTM_pzPose(z_pose_pre, state)
                z_pose_k = reparameterization(
                    z_mu_logvar_pose_k[:, :self.z_dim],
                    z_mu_logvar_pose_k[:, self.z_dim:])
            z_pose_pre = z_pose_k
            z_style_k = torch.empty([bs, self.z_dim]).normal_().cuda()
            theta_k = self.G_pose(z_pose_k)
            c_k, alpha_k = self.G_style(z_style_k)
            x_k, alpha_k = self.STNet(c_k,
                                      theta_k), self.STNet(alpha_k, theta_k)
            X = self.Composer(X, x_k, alpha_k)
        return X.detach().cpu().numpy()

    def visualize(self, img):
        ## one channel to three channels
        # original img
        if img.shape[1] == 1:
            output = img.repeat([1, 3, 1, 1]).cpu().numpy()
        else:
            output = img.cpu().numpy()

        imgs_recons = torch.zeros_like(img)
        D_k = img

        ## latent state for lstm
        state = None

        for k in range(self.num_slots):
            ## inference z_pose from img. q(z_i|z_<i,X)
            if k == 0:
                h = self.E_pose(D_k)
                z_mu_pose_k = h[:, :self.z_dim]
                z_logvar_pose_k = h[:, self.z_dim:]
                z_pose_k = reparameterization(z_mu_pose_k, z_logvar_pose_k)
            else:
                h = self.E_pose(D_k)
                z_mu_logvar_pose_k, state = self.LSTM_qzPose(
                    torch.cat([h, z_pose_pre], dim=1), state)
                z_mu_pose_k = z_mu_logvar_pose_k[:, :self.z_dim]
                z_logvar_pose_k = z_mu_logvar_pose_k[:, self.z_dim:]
                z_pose_k = reparameterization(z_mu_pose_k, z_logvar_pose_k)

            ## store last z_pose for lstm
            z_pose_pre = z_pose_k

            ## inference z_style from img and z_pose. q(z_style_i|z_pose_i,X)
            theta_k = self.G_pose(z_pose_k)
            theta_k_inv = self.InverseTheta(theta_k)
            D_k_canonical = self.STNet(D_k, theta_k_inv)
            z_mu_style_k, z_logvar_style_k = self.E_style(D_k_canonical)
            z_style_k = reparameterization(z_mu_style_k, z_logvar_style_k)

            ## canonical
            C_k, alpha_k = self.G_style(z_style_k)
            L_k, alpha_k = self.STNet(C_k,
                                      theta_k), self.STNet(alpha_k, theta_k)

            ## add box
            if C_k.shape[1] == 1:
                C_k = C_k.repeat([1, 3, 1, 1])
            C_k[:, 0, 0, :] = 1.0
            C_k[:, 0, 63, :] = 1.0
            C_k[:, 0, :, 0] = 1.0
            C_k[:, 0, :, 63] = 1.0

            ## reconstruction with box
            L_k_box = self.STNet(C_k, theta_k)
            output = np.concatenate(
                [output, L_k_box.detach().cpu().numpy()], 0)
            imgs_recons = self.Composer(imgs_recons, L_k_box, alpha_k)
            D_k = F.relu(D_k - L_k)

        output = np.concatenate(
            [output, imgs_recons.detach().cpu().numpy()], 0)

        return output

    @staticmethod
    def get_ema(new, old, alpha):
        if old is None:
            return new
        return (1.0 - alpha) * new + alpha * old

    @staticmethod
    def geco_beta_update(beta,
                         llk_ema,
                         goal,
                         step_size,
                         min_clamp=1e-10,
                         speedup=None):
        # Compute current constraint value and detach because we do not want to
        # back-propagate through error_ema
        constraint = -1.0 * (goal - llk_ema).detach()
        # Update beta
        if speedup is not None and constraint.item() > 0.0:
            # Apply a speedup factor to recover more quickly from undershooting
            beta = beta * torch.exp(speedup * step_size * constraint)
        else:
            beta = beta * torch.exp(step_size * constraint)
        # Clamp beta to be larger than minimum value
        if min_clamp is not None:
            beta = torch.max(beta, torch.tensor(min_clamp).cuda())
        # Detach again just to be safe
        return beta.detach()


if __name__ == "__main__":

    device = torch.device("cuda:0")
    model0 = CST_VAE_lstm(image_channel=1).to(device)
    model = CST_VAE(image_channel=1).to(device)

    img = torch.empty([2, 1, 64, 64]).normal_().to(device)
    X, _, _, _, _, _, _ = model0.forward_path(img)
    loss = model0.loss_function(img)
    model0.eval()
    new_img = model0.generation(3)
    output = model0.visualize(img)
    #plt.imshow(X.detach().permute([0, 2, 3, 1]).numpy()[0])
    #plt.savefig("test.png")
    print(X.shape)
    print(loss)
    print(new_img.shape)
    print(output.shape)
