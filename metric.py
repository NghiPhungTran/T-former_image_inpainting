import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_fid.fid_score import calculate_fid_given_paths
import argparse
import torch

def load_paired_images(gt_dir, gen_dir):
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('_truth.png')]
    gen_files = [f for f in os.listdir(gen_dir) if f.endswith('_out.png')]

    # Lấy tên gốc (không chứa hậu tố _truth / _out)
    gt_basenames = set(f.replace('_truth.png', '') for f in gt_files)
    gen_basenames = set(f.replace('_out.png', '') for f in gen_files)
    common_names = sorted(list(gt_basenames & gen_basenames))

    gt_images = []
    gen_images = []

    for name in common_names:
        gt_path = os.path.join(gt_dir, f"{name}_truth.png")
        gen_path = os.path.join(gen_dir, f"{name}_out.png")

        try:
            gt_img = Image.open(gt_path).convert("RGB").resize((256, 256))
            gen_img = Image.open(gen_path).convert("RGB").resize((256, 256))

            gt_images.append(np.array(gt_img))
            gen_images.append(np.array(gen_img))
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

    return gt_images, gen_images

def compute_psnr_ssim(gt_images, gen_images):
    psnr_list = []
    ssim_list = []
    for gt, gen in zip(gt_images, gen_images):
        if gt.shape != gen.shape:
            print("⚠️ Shape mismatch. Skipping one image pair.")
            continue
        psnr = peak_signal_noise_ratio(gt, gen, data_range=255)
        ssim = structural_similarity(gt, gen, channel_axis=-1, data_range=255)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    if len(psnr_list) == 0 or len(ssim_list) == 0:
        return float('nan'), float('nan')

    return np.mean(psnr_list), np.mean(ssim_list)

def main(gt_dir, gen_dir):
    print(" Loading paired images...")
    gt_images, gen_images = load_paired_images(gt_dir, gen_dir)

    print(" Calculating PSNR & SSIM...")
    psnr, ssim = compute_psnr_ssim(gt_images, gen_images)
    print(f"PSNR: {psnr:.4f}")
    print(f"SSIM: {ssim:.4f}")

    print(" Calculating FID...")
    try:
        fid = calculate_fid_given_paths(
            [gt_dir, gen_dir],
            batch_size=50,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dims=2048
        )
        print(f"FID: {fid:.4f}")
    except Exception as e:
        print(f"FID calculation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground truth images folder')
    parser.add_argument('--gen_dir', type=str, required=True, help='Path to generated (output) images folder')
    args = parser.parse_args()

    main(args.gt_dir, args.gen_dir)