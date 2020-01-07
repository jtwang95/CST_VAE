from torch.utils.data import Dataset
from skimage.transform import rescale, rotate
from skimage import io
import numpy as np
import torch
import pickle
"""
Pytorch dataset class for loading pre-generated images from the CLEVR dataset
"""


class ClevrDataset(Dataset):
    def __init__(self,
                 datapath,
                 data_type='train',
                 max_num_samples=60000,
                 crop_sz=256,
                 down_sz=64):
        suffix = data_type
        self.datapath = datapath + '/CLEVR_' + suffix + '_'
        self.max_num_samples = max_num_samples
        self.crop_sz = crop_sz
        self.down_scale = down_sz / crop_sz

    def __len__(self):
        return self.max_num_samples

    def __getitem__(self, idx):
        imgname = self.datapath + str(idx).zfill(6)
        imgpath = imgname + '.png'
        scaled_img = self.rescale_img(io.imread(imgpath))
        img = torch.tensor(scaled_img, dtype=torch.float32).permute((2, 0, 1))
        return img

    def rescale_img(self, img):
        H, W, C = img.shape
        dH = abs(H - self.crop_sz) // 2
        dW = abs(W - self.crop_sz) // 2
        crop = img[dH:-dH, dW:-dW, :3]
        down = rescale(crop,
                       self.down_scale,
                       order=3,
                       mode='reflect',
                       multichannel=True)
        return down


class MultiMnistDataset(Dataset):
    def __init__(self, datapath, data_type='common',data_name='2'):
        suffix = data_type
        self.filename = datapath + suffix + data_name + '.pkl'
        self.data = pickle.load(open(self.filename, "rb"))

    def __len__(self):
        return self.data["image"].shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(np.expand_dims(self.data["image"][idx], 2),
                           dtype=torch.float32).permute((2, 0, 1))
        return img


def main():
    import matplotlib.pyplot as plt
    data_path_mm = './multi_mnist_data/'
    d = MultiMnistDataset(data_path_mm,data_name='5')
    data = torch.utils.data.DataLoader(d,
                                       batch_size=16,
                                       shuffle=False,
                                       num_workers=1)
    fig, ax = plt.subplots(4, 4)
    for i, batch in enumerate(data):
        if i > 1: break
        #print('On batch {}'.format(i))
        for j in range(batch.shape[0]):
            #print(batch.shape[0])
            row = j // 4
            col = j % 4
            ax[row, col].axis("off")
            ax[row, col].imshow(batch[j].permute(
                (1, 2, 0)).detach().numpy().squeeze())
        #print(type(x))
        #plt.imshow(x.permute((1, 2, 0)).detach().numpy().squeeze())

    plt.savefig("test.png")


if __name__ == '__main__':
    main()
