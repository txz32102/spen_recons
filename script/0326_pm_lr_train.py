""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=1 python3 script/0125_pm_lr_train.py
"""


import argparse
import os
from datetime import datetime
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from model.simple_gan_0125 import Generator, Discriminator
from dataset.spen_dataset_0125 import SpenDataset
from util.logger_0125 import Logger
from util.physical_model_0125 import physical_model

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def _complex_to_1ch(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex tensor (B,1,W,H) -> real tensor (B,1,W,H).
    Uses magnitude, normalizes to [0,1], then rescales to [-1,1].
    """
    if not torch.is_complex(x):
        raise ValueError("Input must be a complex tensor")

    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected input of shape (B,1,W,H), got {tuple(x.shape)}")

    # Magnitude
    mag = x.abs()  # (B,1,W,H), real

    # Normalize to [0,1]
    max_val = mag.amax(dim=(2,3), keepdim=True)
    normed = mag  / max_val

    # Rescale to [-1,1]
    out = normed * 2.0 - 1.0
    return out


def set_seed(seed=0):
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch CPU & GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    
    # CuDNN determinism (Note: This might slightly impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

time_prefix = datetime.now().strftime("%m%d%H%M")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/0125_mixed_2000', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=96, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='disable GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--log_dir', type=str, default=f'log/{time_prefix}_pm_lr', help='directory to save logs and model checkpoints')
parser.add_argument('--ckpt_save_freq', type=int, default=50, help='save checkpoint frequency (in epochs)')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# --- nets ---
netG = Generator(opt.input_nc, opt.output_nc)
netD_hr = Discriminator(opt.output_nc)   # disc over HR (1ch)
netD_lr = Discriminator(opt.output_nc)   # disc over LR (1ch mag)

PM = physical_model()  # HR->[complex LR]; we will convert to 1ch

if opt.cuda:
    netG.cuda(); netD_hr.cuda(); netD_lr.cuda()

netG.apply(weights_init_normal)
netD_hr.apply(weights_init_normal)
netD_lr.apply(weights_init_normal)

# --- losses & opts ---
criterion_GAN   = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

optimizer_G    = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_hr = torch.optim.Adam(netD_hr.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_lr = torch.optim.Adam(netD_lr.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G    = torch.optim.lr_scheduler.LambdaLR(optimizer_G,    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_hr = torch.optim.lr_scheduler.LambdaLR(optimizer_D_hr, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_lr = torch.optim.lr_scheduler.LambdaLR(optimizer_D_lr, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# --- helpers ---
device = torch.device("cuda" if opt.cuda else "cpu")


# targets shaped like D outputs (B,1)
target_real = torch.ones((opt.batchSize, 1), dtype=torch.float32, device=device)
target_fake = torch.zeros((opt.batchSize, 1), dtype=torch.float32, device=device)

fake_hr_buf = ReplayBuffer()
fake_lr_buf = ReplayBuffer()

dataloader = DataLoader(SpenDataset(opt.dataroot, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True,
                        num_workers=opt.n_cpu, drop_last=True)

os.makedirs(opt.log_dir, exist_ok=True)
os.makedirs(f'{opt.log_dir}/train', exist_ok=True)
logger = Logger(opt.n_epochs, len(dataloader), f'{opt.log_dir}/train')

# --- training ---
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_hr_11 = batch['hr'].to(device)         # HR in [-1,1], 1ch
        real_lr_11 = batch['lr'].to(device)         # LR may be 1ch or 2ch

        # --------- G step ---------
        optimizer_G.zero_grad()

        # path A: HR -> PM -> LR1 -> G -> HR_hat ; adversarial on HR
        pm_lr_1 = PM((real_hr_11+1)/2)             # 1ch [-1,1]
        pm_lr_1 = _complex_to_1ch(pm_lr_1)
        fake_hr = netG(pm_lr_1)                     # 1ch [-1,1]
        pred_fake_hr = netD_hr(fake_hr)
        loss_GAN_hr = criterion_GAN(pred_fake_hr, target_real)

        # path B: LR (data) -> G -> HR_tilde -> PM -> LR_tilde ; adversarial on LR
        recovered_hr = netG(real_lr_11)             # 1ch [-1,1]
        pm_lr_2 = PM((recovered_hr+1)/2)        # 1ch [-1,1]
        pm_lr_2 = _complex_to_1ch(pm_lr_2)
        pred_fake_lr = netD_lr(pm_lr_2)
        loss_GAN_lr = criterion_GAN(pred_fake_lr, target_real)

        # cycle losses (both in matching spaces)
        # LR->HR->PM should match LR (both 1ch [-1,1])
        loss_cycle_lrhrlr  = criterion_cycle(pm_lr_2, real_lr_11) * 5.0
        # HR->PM->G should reconstruct HR
        loss_cycle_hrlrhr  = criterion_cycle(fake_hr, real_hr_11) * 5.0

        loss_G = loss_GAN_hr + loss_GAN_lr + loss_cycle_lrhrlr + loss_cycle_hrlrhr
        loss_G.backward()
        optimizer_G.step()

        # --------- D_hr step ---------
        optimizer_D_hr.zero_grad()
        pred_real_hr = netD_hr(real_hr_11)
        loss_D_hr_real = criterion_GAN(pred_real_hr, target_real)

        fake_hr_buf_out = fake_hr_buf.push_and_pop(fake_hr.detach())
        pred_fake_hr = netD_hr(fake_hr_buf_out)
        loss_D_hr_fake = criterion_GAN(pred_fake_hr, target_fake)

        loss_D_hr = 0.5 * (loss_D_hr_real + loss_D_hr_fake)
        loss_D_hr.backward()
        optimizer_D_hr.step()

        # --------- D_lr step ---------
        optimizer_D_lr.zero_grad()
        pred_real_lr = netD_lr(real_lr_11)
        loss_D_lr_real = criterion_GAN(pred_real_lr, target_real)

        fake_lr_buf_out = fake_lr_buf.push_and_pop(pm_lr_2.detach())
        pred_fake_lr = netD_lr(fake_lr_buf_out)
        loss_D_lr_fake = criterion_GAN(pred_fake_lr, target_fake)

        loss_D_lr = 0.5 * (loss_D_lr_real + loss_D_lr_fake)
        loss_D_lr.backward()
        optimizer_D_lr.step()

        # --------- logging ---------
        if i == len(dataloader) - 1:
            logger.log(
                {
                    'loss_G': loss_G,
                    'loss_GAN_hr': loss_GAN_hr,
                    'loss_GAN_lr': loss_GAN_lr,
                    'loss_cycle_lrhrlr': loss_cycle_lrhrlr,
                    'loss_cycle_hrlrhr': loss_cycle_hrlrhr,
                    'loss_D_hr': loss_D_hr,
                    'loss_D_lr': loss_D_lr
                },
                images={
                    'real_hr': real_hr_11,
                    'real_lr': real_lr_11,
                    'fake_hr': fake_hr,
                    'pm_lr_from_recHR': pm_lr_2,
                    'pm_lr_from_realHR': pm_lr_1,
                    'recovered_hr': recovered_hr
                }
            )
        else:
            logger.log(
                {
                    'loss_G': loss_G,
                    'loss_GAN_hr': loss_GAN_hr,
                    'loss_GAN_lr': loss_GAN_lr,
                    'loss_cycle_lrhrlr': loss_cycle_lrhrlr,
                    'loss_cycle_hrlrhr': loss_cycle_hrlrhr,
                    'loss_D_hr': loss_D_hr,
                    'loss_D_lr': loss_D_lr
                }
            )

    # schedulers
    lr_scheduler_G.step()
    lr_scheduler_D_hr.step()
    lr_scheduler_D_lr.step()

    # checkpoints
    if (epoch % opt.ckpt_save_freq) == 0:
        wdir = f'{opt.log_dir}/weights'; os.makedirs(wdir, exist_ok=True)
        torch.save(netG.state_dict(),     f'{wdir}/netG_lr2hr.pth')
        torch.save(netD_hr.state_dict(),  f'{wdir}/netD_hr.pth')
        torch.save(netD_lr.state_dict(),  f'{wdir}/netD_lr.pth')
        print(f"[Checkpoint] Saved models at epoch {epoch}")