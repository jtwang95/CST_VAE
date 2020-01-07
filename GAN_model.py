from networks import *


class Layer_Generator(nn.Module):
    def __init__(self,
                 batch_size=32,
                 image_channel=3,
                 canvas_size=64,
                 z_dim=128):
        super().__init__()
        self.canvas_size = canvas_size
        self.z_dim = z_dim
        self.image_channel = image_channel
        self.batch_size = batch_size

        ## basic modules
        self.G_pose = Generator_Pose()
        self.G_style = Generator_Style()
        self.STNet = STNet()

    def forward(self):
        z_pose = torch.empty([self.batch_size, self.z_dim]).normal_()
        theta = self.G_pose(z_pose)
        z_style = torch.empty([self.batch_size, self.z_dim]).normal_()
        x, alpha = self.G_style(z_style)
        return self.STNet(x, theta), alpha, [x, theta, z_style, z_pose]


class Image_Generator(nn.Module):
    def __init__(self,
                 num_slots,
                 batch_size=32,
                 image_channel=3,
                 canvas_size=64,
                 z_dim=128):
        super().__init__()
        self.canvas_size = canvas_size
        self.z_dim = z_dim
        self.image_channel = image_channel
        self.batch_size = batch_size
        self.num_slots = num_slots

        ## basic modules
        self.OneLayer_Generator = Layer_Generator(
            batch_size=self.batch_size,
            image_channel=self.image_channel,
            canvas_size=self.canvas_size,
            z_dim=self.z_dim)
        self.Composer = Compose()

    def forward(self):
        X = torch.empty([
            self.batch_size, self.image_channel, self.canvas_size,
            self.canvas_size
        ]).zero_()
        x_list = []
        theta_list = []
        z_pose_list = []
        z_style_list = []

        for k in range(self.num_slots):
            output = self.OneLayer_Generator()
            print(output[1].shape)
            X = self.Composer(X, output[0], output[1])
            x_list.append(output[2][0])
            theta_list.append(output[2][1])
            z_style_list.append(output[2][2])
            z_pose_list.append(output[2][3])

        return X, x_list, theta_list, z_style_list, z_pose_list


if __name__ == "__main__":
    G_Image = Image_Generator(2)
    X, _, _, _, _ = G_Image()
    plt.imshow(X.detach().permute([0, 2, 3, 1]).numpy()[0])
    plt.savefig("test.png")