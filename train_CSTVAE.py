import os

import torch
import torch.optim as optim
from Utils.checkpoints import save_context
from Utils.checkpoints import plot_image
from Utils import flags, range_iterator
import matplotlib.pyplot as plt
import numpy as np

from data.datasets import MultiMnistDataset
from Torture.dataset import CLEVRDataset
from VAE_model import *

DATA_NAME = '3'
EVALUATE_EPOCH = 1
SAVE_EPOCH = 10
EPOCH_TOTAL = 300
VISUALIZE_EVERY = 500
SUMMARY_EVERY = 1000
HYPERPARAMETERS = None
DEFAULT_RESULTS_FOLDER_ARGUMENT = "Not Valid"
DEFAULT_RESULTS_FOLDER = "./results/"
FILES_TO_BE_SAVED = ["./", "./data/"]
KEY_ARGUMENTS = ["model"]
config = {
    "DEFAULT_RESULTS_FOLDER": DEFAULT_RESULTS_FOLDER,
    "FILES_TO_BE_SAVED": FILES_TO_BE_SAVED,
    "KEY_ARGUMENTS": KEY_ARGUMENTS
}

flags.DEFINE_argument("-gpu", "--gpu", default="-1")
flags.DEFINE_argument("--results-folder",
                      default=DEFAULT_RESULTS_FOLDER_ARGUMENT)
flags.DEFINE_argument("-k", "-key", "--key", default="")
flags.DEFINE_argument("-data", "--data", default="multimnist")
flags.DEFINE_boolean("-o", "--overwrite-results", default=False)
flags.DEFINE_argument("-bs",
                      "-batch_size",
                      "--batch_size",
                      type=int,
                      default=32)
flags.DEFINE_argument("-is",
                      "-image_size",
                      "--image_size",
                      type=int,
                      default=64)
flags.DEFINE_argument("-ic",
                      "-image_channel",
                      "--image_channel",
                      type=int,
                      default=3)
flags.DEFINE_argument("-nw",
                      "-num_workers",
                      "--num_workers",
                      type=int,
                      default=64)
flags.DEFINE_argument("-tp",
                      "-theta_parametrization",
                      "--theta_parametrization",
                      type=str,
                      default="222")
flags.DEFINE_argument("-ns", "-num_slot", "--num_slot", type=int, default=2)
flags.DEFINE_argument("-model", "--model", default="cst_vae")
flags.DEFINE_argument("-mode", "--mode", default="train")
flags.DEFINE_argument("-oldmodel", "--oldmodel", default="")
FLAGS = flags.FLAGS

logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, FLAGS, config)
logger.info("build dataloader")

if FLAGS.data.lower() in ["clevr", "clever"]:
    trainset = CLEVRDataset(phase='train', size=FLAGS.image_size)
    testset = CLEVRDataset(phase='test', size=FLAGS.image_size)
    IMAGE_CHANNEL = 3
elif FLAGS.data.lower() in ["multimnist", "mm"]:
    trainset = MultiMnistDataset(
        datapath="/home/jitao/scripts/CST_VAE/data/multi_mnist_data/",
        data_type="common",
        data_name=DATA_NAME)
    testset = MultiMnistDataset(
        datapath="/home/jitao/scripts/CST_VAE/data/multi_mnist_data/",
        data_type="test",
        data_name=DATA_NAME)
    IMAGE_CHANNEL = 1

dataloader_train = torch.utils.data.DataLoader(trainset,
                                               batch_size=FLAGS.batch_size,
                                               shuffle=True,
                                               num_workers=FLAGS.num_workers,
                                               drop_last=True)
dataloader_test = torch.utils.data.DataLoader(testset,
                                              batch_size=10,
                                              shuffle=True,
                                              num_workers=FLAGS.num_workers,
                                              drop_last=False)
iterator_length = len(dataloader_train)
logger.info("{} iterations per Epoch, check it!".format(iterator_length))

device = torch.device("cuda:0")
if FLAGS.model in ['cst_vae']:
    model = CST_VAE(
        num_slots=FLAGS.num_slot,
        canvas_size=FLAGS.image_size,
        image_channel=IMAGE_CHANNEL,
        beta=0.5,
        gamma=0.5,
        theta_parametrization=FLAGS.theta_parametrization).to(device)


def anneal_lr(epoch):
    if epoch < 100:
        return 1.
    elif epoch < 150:
        return 0.1
    else:
        return 0.01


optimizer = optim.Adam(model.parameters(), lr=0.0001)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, [anneal_lr])


def visualize_test(name):
    if isinstance(name, int):
        base_format = "epoch_{:06d}_{}"
    else:
        base_format = "epoch_{}_{}"

    with torch.no_grad():
        model.eval()
        visualize_img = next(iter(dataloader_test))
        visualize_img = model.visualize(visualize_img.to(device))
        plot_image(visualize_img,
                   os.path.join(SUMMARIES_FOLDER,
                                base_format.format(name, "reconstruction")),
                   shape=[2 + FLAGS.num_slot, 10],
                   figsize=[10 * 5, (2 + FLAGS.num_slot) * 5])
        model.train()


def visualize_gen(name):
    if isinstance(name, int):
        base_format = "epoch_{:06d}_{}"
    else:
        base_format = "epoch_{}_{}"

    with torch.no_grad():
        model.eval()
        visualize_img = model.generation(bs=10)
        plot_image(visualize_img,
                   os.path.join(SUMMARIES_FOLDER,
                                base_format.format(name, "generation")),
                   shape=[2, 5],
                   figsize=[5 * 5, 2 * 5])
        model.train()


if FLAGS.mode != "train":
    model.load_state_dict(torch.load(FLAGS.oldmodel))
    with torch.no_grad():
        model.eval()
        visualize_test("test")
        visualize_gen("test")
    exit()

iters = 0
for epoch in range(EPOCH_TOTAL):  # loop over the dataset multiple times
    logger.info("Start Epoch {}".format(epoch))
    running_loss = 0.0

    model.train()
    for img_batch in range_iterator(dataloader_train):
        img_batch = img_batch.to(device)
        #print(img_batch.max(), img_batch.min())
        # print(img_batch.size())
        model.zero_grad()
        loss = model.loss_function(img_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iters += 1
        if iters % VISUALIZE_EVERY == 0:
            if iters % SUMMARY_EVERY == 0:
                img_name = iters
            else:
                img_name = "0latest"
            # model.eval()
            # img_gen = model.generation()
            # plt.imshow(
            #     np.squeeze(img_gen.detach().cpu().permute([0, 2, 3,
            #                                                1]).numpy()))
            # plt.savefig("Fig0latest" + ".png")
            # model.train()

            visualize_test(img_name)
            visualize_gen(img_name)

    possible_lr = []
    for param_group in optimizer.param_groups:
        possible_lr.append(param_group['lr'])
    lr_scheduler.step()
    logger.info('[{}] train loss: {:.4f}, learning rate: {}'.format(
        epoch + 1, running_loss / iterator_length, str(possible_lr)))

    if epoch % SAVE_EPOCH == 0 or epoch == EPOCH_TOTAL - 1:
        torch.save(model.state_dict(),
                   os.path.join(MODELS_FOLDER, "epoch{}.ckpt".format(epoch)))

logger.info('Finished Training')
