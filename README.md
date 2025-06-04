# T-former: An Efficient Transformer for Image Inpainting (MM 2022)
This is the code for ACM multimedia 2022 “T-former: An Efficient Transformer for Image Inpainting”
# visualization during training
python -m visdom.server
# train:
python train.py --no_flip --no_rotation --no_augment --img_file ./data_split/train --val_img_file ./data_split/val --lr 1e-4 --batchSize 1 --niter 300000 
# fine_tune:
python train.py --no_flip --no_rotation --no_augment --img_file ./data_split/train --val_img_file ./data_split/val --lr 1e-5 --batchSize 1 --niter 380000 --continue_train --which_iter 200000 
# test:
python test.py --batchSize 1 --mask_type 2 --img_file ./data_split/test --mask_file none
# evaluation:
python evaluation.py --folder ./result/paris

## Citation
If you are interested in this work, please consider citing:

    @inproceedings{tformer_image_inpainting,
      author = {Deng, Ye and Hui, Siqi and Zhou, Sanping and Meng, Deyu and Wang, Jinjun},
      title = {T-former: An Efficient Transformer for Image Inpainting},
      year = {2022},
      isbn = {9781450392037},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      doi = {10.1145/3503161.3548446},
      pages = {6559–6568},
      numpages = {10},
      series = {MM '22}
}

