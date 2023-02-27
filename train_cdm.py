import os
import torch
import argparse
from omegaconf import OmegaConf
from utils import instantiate_from_config, save_img
from pprint import pprint
import warnings
from ning.logger import create_logger
import random
import datetime

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='us image generation parameters')

parser.add_argument(
    "-b",
    "--base",
    nargs="*",
    metavar="base_config.yaml",
    help="paths to base configs. Loaded from left-to-right. "
    "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    default=list(),
)

parser.add_argument("-e",
                    "--num-epochs",
                    help="num of epochs",
                    type=int,
                    default=1000)

parser.add_argument("-g", "--gpu", help="num of epochs", type=int, default=0)

opt = parser.parse_args()
pprint(opt)
device = torch.device('cuda:{}'.format(opt.gpu))

config = OmegaConf.load(opt.base[0])

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
opt.name = opt.base[0].split('/')[-1].split('.')[0]
nowname = now + "_" + opt.name
logdir = os.path.join("logs", nowname)
ckptdir = os.path.join(logdir, "checkpoints")
imgdir = os.path.join(logdir, "images/val")
os.makedirs(logdir)
os.mkdir(ckptdir)
os.makedirs(imgdir)

logger = create_logger(output_dir=logdir, name='')


data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()
train_dataloader = data.train_dataloader()
val_dataloader = data.val_dataloader()

model = instantiate_from_config(config.model)
model.set_device(device)

ngpu = 1
accumulate_grad_batches = 1
bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
optimizer = model.configure_optimizers()


global_step = 0
for epoch in range(opt.num_epochs):

    # model.train()
    # for batch_idx, batch in enumerate(train_dataloader):
    #     loss = model.training_step(batch=batch, batch_idx=batch_idx)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if batch_idx % 10 == 0:
    #         train_info = '[Train] Epoch: {}, mse_loss: {:.4f}'.format(epoch, loss.cpu().item())
    #         logger.info(train_info)

    #     global_step += 1

    model.eval()
    save_config = dict()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            save_config['flag'] = True
            save_config['split'] = 'val'
            save_config['epoch'] = epoch
            save_config['global_step'] = global_step
            save_config['batch_index'] = batch_idx
            save_config['imgdir'] = imgdir

            loss, loss_dict = model.validation_step(batch=batch, batch_idx=batch_idx, save_config=save_config)

            if batch_idx % 10 == 0:
                val_info = '[Validation] Epoch: {}, mse_loss: {:.4f}, '.format(epoch, loss.cpu().item()) + str(loss_dict)
                # val_info = '[Validation] Epoch: {}, mse_loss: '.format(epoch) + str(loss_dict_ema)
                logger.info(val_info)