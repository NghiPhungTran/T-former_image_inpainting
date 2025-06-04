import numpy as np
import argparse
from PIL import Image
import math
import os
import glob

parser = argparse.ArgumentParser(description='Evaluation on the dataset')
parser.add_argument('--save_path', type=str, required=True,
                    help='Path to folder containing *_out.png and *_truth.png')
parser.add_argument('--num_test', type=int, default=1000,
                    help='Number of test images to evaluate')
parser.add_argument('--resize', type=int, nargs=2, default=[256, 256],
                    help='Resize all images to this size, e.g. --resize 256 256')
args = parser.parse_args()


def compute_errors(gt, pre):
    l1 = np.mean(np.abs(gt - pre))

    mse = np.mean((gt - pre) ** 2)
    PSNR = 100 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))

    gx = pre - np.roll(pre, -1, axis=1)
    gy = pre - np.roll(pre, -1, axis=0)
    grad_norm2 = gx ** 2 + gy ** 2
    TV = np.mean(np.sqrt(grad_norm2))

    return l1, PSNR, TV


from itertools import chain

def find_all_images(path, suffixes):
    return sorted(list(chain.from_iterable(glob.glob(os.path.join(path, f'*_out.{ext}')) for ext in suffixes)))

if __name__ == "__main__":
    suffixes = ['png', 'jpg', 'jpeg']
    out_paths = sorted(list(chain.from_iterable(glob.glob(os.path.join(args.save_path, f'*_out.{s}')) for s in suffixes)))
    truth_paths = sorted(list(chain.from_iterable(glob.glob(os.path.join(args.save_path, f'*_truth.{s}')) for s in suffixes)))

    if len(out_paths) == 0 or len(truth_paths) == 0:
        raise FileNotFoundError("Không tìm thấy ảnh *_out.(png/jpg/jpeg) hoặc *_truth.(png/jpg/jpeg) trong thư mục.")


    if len(out_paths) != len(truth_paths):
        raise ValueError(f"Số lượng ảnh không khớp: {len(out_paths)} _out vs {len(truth_paths)} _truth")

    total = min(args.num_test, len(out_paths))
    print(f"Đang đánh giá {total} ảnh...")

    l1_iter = np.zeros(total, np.float32)
    PSNR_iter = np.zeros(total, np.float32)
    TV_iter = np.zeros(total, np.float32)

    for j in range(total):
        pre_image = Image.open(out_paths[j]).resize(args.resize).convert('RGB')
        gt_image = Image.open(truth_paths[j]).resize(args.resize).convert('RGB')

        pre_numpy = np.array(pre_image).astype(np.float32)
        gt_numpy = np.array(gt_image).astype(np.float32)

        l1_temp, PSNR_temp, TV_temp = compute_errors(gt_numpy, pre_numpy)
        l1_iter[j], PSNR_iter[j], TV_iter[j] = l1_temp, PSNR_temp, TV_temp

        print(f'[{j+1}/{total}] {os.path.basename(out_paths[j])} -> '
              f'L1: {l1_temp:.4f}, PSNR: {PSNR_temp:.4f}, TV: {TV_temp:.4f}')

    print('\n===== Tổng kết =====')
    print('{:>10},{:>10},{:>10}'.format('L1_LOSS', 'PSNR', 'TV'))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(l1_iter.mean(), PSNR_iter.mean(), TV_iter.mean()))
