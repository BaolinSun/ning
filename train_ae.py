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
                    default=100000)

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
# test_dataloader = data.test_dataloader

model = instantiate_from_config(config.model).to(device)
model.loss = model.loss.to(device)

ngpu = 1
accumulate_grad_batches = 1
bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr

[opt_ae, opt_disc] = model.configure_optimizers()

rec_loss_top3 = [99999]*3
global_step = 0
for epoch in range(opt.num_epochs):
    model.train()
    train_info = ''
    batch_index = 0
    save_config = dict()
    save_index = random.randint(1, int(len(train_dataloader)/config.data.params.batch_size))
    for batch in train_dataloader:

        optimizer_idx = global_step % 2
        loss, log_dict = model.training_step(batch=batch,
                                             optimizer_idx=optimizer_idx,
                                             global_step=int(global_step / 2),
                                             device=device,
                                             save_config=save_config)

        if optimizer_idx == 0:
            opt_ae.zero_grad()
            loss.backward()
            opt_ae.step()

            train_info += '[Train] Epoch: {}, aeloss: {:.4f}, rec_loss: {:.4f}'.format(
                epoch, loss.cpu().item(), log_dict['train/rec_loss'].cpu().item())

        elif optimizer_idx == 1:
            opt_disc.zero_grad()
            loss.backward()
            opt_disc.step()

            if (global_step+1) % 10 == 0:
                train_info += ' disc_loss: {:.4f}, logits_real: {:.4f}, logits_fake: {:.4f}'.format(
                    log_dict['train/disc_loss'].cpu().item(),
                    log_dict['train/logits_real'].cpu().item(),
                    log_dict['train/logits_fake'].cpu().item())
                logger.info(train_info)
            train_info = ''

        global_step += 1
        batch_index += 1


    model.eval()
    save_config = dict()
    with torch.no_grad():
        rec_loss_mean = 0
        save_flag = 0
        batch_index = 0
        save_index = random.randint(1, int(len(val_dataloader)/config.data.params.batch_size))
        for batch_idx, batch in enumerate(val_dataloader):
            save_config['flag'] = True
            save_config['split'] = 'val'
            save_config['epoch'] = epoch
            save_config['global_step'] = global_step
            save_config['batch_index'] = batch_idx
            save_config['imgdir'] = imgdir

            aeloss, rec_loss = model.validation_step(batch, global_step, device, save_config)


            if batch_index % 10 == 0:
                val_info = '[Validation] Epoch: {}, aeloss: {:.4f}, rec_loss: {:.4f}'.format(
                    epoch, aeloss.cpu().item(), rec_loss.cpu().item())
                logger.info(val_info)

            rec_loss_mean += rec_loss.cpu().item()


        
        rec_loss_mean = rec_loss_mean / len(val_dataloader)
        save_cpt_flag = 0
        if rec_loss_mean < rec_loss_top3[2]:
            rec_loss_top3[0] = rec_loss_top3[1]
            rec_loss_top3[1] = rec_loss_top3[2]
            rec_loss_top3[2] = rec_loss_mean
            save_cpt_flag = 1
        elif rec_loss_mean < rec_loss_top3[1]:
            rec_loss_top3[0] = rec_loss_top3[1]
            rec_loss_top3[1] = rec_loss_mean
            save_cpt_flag = 2
        elif rec_loss_mean < rec_loss_top3[0]:
            rec_loss_top3[0] = rec_loss_mean
            save_cpt_flag = 3

        if save_cpt_flag != 0:
            logger.info('current res loss: {:.4f} in top-3, best rec loss: [{:.4f}], saving model'.format(rec_loss_mean, rec_loss_top3[2]))            
            torch.save(model.state_dict(), ckptdir+"/checkpoints_top{}.pth".format(save_cpt_flag))





        
