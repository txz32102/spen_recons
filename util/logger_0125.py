import os
import sys
import time
import datetime
import numpy as np
from PIL import Image

def tensor2image(tensor):
    """
    Convert a torch tensor in [-1,1] to a grayscale numpy image.
    Handles:
      (1,H,W) -> (H,W)
      (2,H,W) -> sqrt(real^2 + imag^2) -> (H,W)
    """
    t = tensor.detach()
    arr = t[0].cpu().float().numpy()  # (C,H,W)

    if arr.shape[0] == 1:
        # Single channel -> grayscale
        img = arr[0]
        img = 127.5 * (img + 1.0)
    elif arr.shape[0] == 2:
        # Two channels -> magnitude
        real, imag = arr[0], arr[1]
        img = np.sqrt(((real+1)/2)**2 + ((imag+1)/2)**2)
        img = img * 255.0
    else:
        raise ValueError(f"Unsupported channel count: {arr.shape[0]}")

    
    return img.astype(np.uint8)

class Logger:
    def __init__(self, n_epochs, batches_epoch, log_dir='logs', running_avg=False):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0.0

        self.running_avg = running_avg

        self.losses = {}              # per-batch values or running sums (see running_avg)
        self.counts = {}              # only used if running_avg=True
        self.header_written = False   # write header lazily on first log()

        # dirs
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.losses_log_path = os.path.join(log_dir, 'losses.csv')
        self.images_log_path = os.path.join(log_dir, 'images')
        os.makedirs(self.images_log_path, exist_ok=True)

        # create empty file
        with open(self.losses_log_path, 'w') as f:
            pass

    def _ensure_header(self, losses):
        if not self.header_written:
            keys = list(losses.keys())  # use names exactly as passed in
            with open(self.losses_log_path, 'a') as f:
                f.write('epoch,batch,' + ','.join(keys) + '\n')
            self.header_written = True

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        # Write header lazily when we first see losses
        if losses and not self.header_written:
            self._ensure_header(losses)

        # Progress line
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' %
                         (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        # Update tracked losses
        row_vals = []
        if losses:
            for i, (name, val) in enumerate(losses.items()):
                v = float(val.item() if hasattr(val, "item") else val)

                if self.running_avg:
                    # accumulate sums + counts
                    self.losses[name] = self.losses.get(name, 0.0) + v
                    self.counts[name] = self.counts.get(name, 0) + 1
                    disp = self.losses[name] / self.counts[name]
                else:
                    # per-batch value
                    self.losses[name] = v
                    disp = v

                row_vals.append(disp)

                sep = ' -- ' if (i + 1) == len(losses) else ' | '
                sys.stdout.write(f'{name}: {disp:.4f}{sep}')

        # ETA
        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + (self.batches_epoch - self.batch)
        eta = datetime.timedelta(seconds=(batches_left * self.mean_period / max(1, batches_done)))
        sys.stdout.write(f'ETA: {eta}')

        # Append CSV row
        if losses:
            with open(self.losses_log_path, 'a') as f:
                f.write(f'{self.epoch},{self.batch},' +
                        ','.join(f'{x:.6f}' for x in row_vals) + '\n')

        if images:
            for name, t in images.items():
                img = tensor2image(t)
                Image.fromarray(img).save(os.path.join(self.images_log_path, f'{self.epoch}_{self.batch}_{name}.png'))

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            if self.running_avg:
                # reset running stats at epoch end
                self.losses.clear()
                self.counts.clear()
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1