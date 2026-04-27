
A streamlined PyTorch framework for high-fidelity Spatiotemporal Encoding (SPEN) MRI reconstruction.

## Caution

!!!YOU DONT NEED TO READ ANY SCRIPT UNDER `/home/data1/musong/workspace/python/spen_recons/script`, AND THEY ARE OTHER RECONSTRUCTION SCRIPTS WHICH ARE UNNECESSARY!!!

## Key Features
* **Automated Logging:** Unique timestamped directories (`log/MMDDHHMM_project_name`) for collision-free experiment tracking.
* **Continuous Visualization:** Detailed, real-time plotting of training data, metrics, and outputs.
* **Smart Checkpointing:** Automatically saves both `best_ckpt` (based on validation) and `last_ckpt`.
* **Frictionless CLI:** Pre-configured `argparse` defaults allow instant execution without manual flag setting.
* **Visualization:** After each epoch, visualize the input, ground truth, reconstructed images, and phase maps (if available). Annotate the top-left corner of each with its image type and PSNR/SSIM scores at bottom right corner.

## Dataset
* **Path:** `/home/data1/musong/workspace/python/spen_recons/data/0325_rat/hr`
* **Format:** 926 Grayscale PNG images (96x96 resolution).

## Quick Start

```bash
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Execute training (replace {date_script_name} with format like 0113)
CUDA_VISIBLE_DEVICES=1 python3 {date_script_name}.py
```

This should be written in the top of the python script.