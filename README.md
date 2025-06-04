# T-former: An Efficient Transformer for Image Inpainting (MM 2022)
This is the code for ACM multimedia 2022 “T-former: An Efficient Transformer for Image Inpainting”
# visualization during training
python -m visdom.server
# train:
python train.py --no_flip --no_rotation --no_augment --img_file ./data_split/train --lr 1e-4 --batchSize 1 --niter 300000 
# fine_tune:
python train.py --no_flip --no_rotation --no_augment --img_file your_data --lr 1e-5 --continue_train
# test:
python test.py --batchSize 1 --mask_type 2 --img_file ./data_split/test --mask_file none
python test.py --batchSize 1 --mask_type 2 --img_file ./data_split/test --mask_file none 
python train.py --no_flip --no_rotation --no_augment --img_file ./data_split/train --val_img_file ./data_split/val --lr 1e-5 --batchSize 1 --niter 380000 --continue_train --which_iter 200000 

python metric.py --gt_dir ./result/paris/truth_images --gen_dir ./result/paris/out_images
python train.py --no_flip --no_rotation --no_augment --img_file ./data_split/test_100 --val_img_file ./data_split/test_10 --lr 1e-4 --batchSize 1 --niter 1000 --display_id 1
python train.py --no_flip --no_rotation --no_augment --img_file ./data_split/train --val_img_file ./data_split/val --lr 1e-4 --batchSize 1 --niter 300000 
.\venv\Scripts\activate
C:\Users\LOQ\T-former_image_inpainting
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

