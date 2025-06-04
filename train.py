import time
from collections import defaultdict
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model import create_model
from util.visualizer import Visualizer
import torch
import os

if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()

    # create train and val dataloaders
    train_loader, val_loader = dataloader(opt)  # giả sử dataloader trả về 2 loader train và val
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    print(f'Training images = {train_size}, Validation images = {val_size}')

    # create a model
    model = create_model(opt)

    if opt.continue_train:
        total_iteration = int(opt.which_iter)
        model.load_networks(total_iteration)
        epoch = total_iteration // train_size
        print(f"=> Resuming from iteration {total_iteration}, estimated epoch {epoch}")
    else:
        total_iteration = 0
        epoch = 0


    # create a visualizer
    visualizer = Visualizer(opt)

    # training flag
    keep_training = True
    max_iteration = opt.niter + opt.niter_decay

    while keep_training:
        epoch += 1
        print(f'\nTraining epoch: {epoch}')
        epoch_start_time = time.time()

        # -------- TRAIN --------
        model.isTrain = True
        train_loss_total = defaultdict(float)
        train_count = 0

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            total_iteration += 1

            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_errors()  # dict, vd: {'G_loss': 0.1, 'D_loss': 0.2, ...}
            batch_size = data['img'].size(0) if 'img' in data else 1

            for k, v in losses.items():
                train_loss_total[k] += v * batch_size
            train_count += batch_size

            # Display images
            if total_iteration % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            # Print loss info
            if total_iteration % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / batch_size
                visualizer.print_current_errors(epoch, total_iteration, losses, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(total_iteration, losses)

            # Save latest model
            if total_iteration % opt.save_latest_freq == 0:
                print(f'Saving latest model (epoch {epoch}, total_steps {total_iteration})')
                model.save_networks('latest')

            # Save model at iterations freq
            if total_iteration % opt.save_iters_freq == 0:
                print(f'Saving model at iteration {total_iteration}')
                model.save_networks(total_iteration)

            if total_iteration > max_iteration:
                keep_training = False
                break

        avg_train_loss = {k: v / train_count for k, v in train_loss_total.items()}

        # -------- VALIDATION --------
        model.isTrain = False
        val_loss_total = defaultdict(float)
        val_count = 0

        with torch.no_grad():
            for val_data in val_loader:
                model.set_input(val_data)
                model.forward()
                val_losses = model.get_current_errors()
                batch_size = val_data['img'].size(0) if 'img' in val_data else 1

                for k, v in val_losses.items():
                    val_loss_total[k] += v * batch_size
                val_count += batch_size

        avg_val_loss = {k: v / val_count for k, v in val_loss_total.items()}

        # -------- LOGGING --------
        print(f'\nEpoch {epoch} Summary:')
        log_file_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        all_loss_keys = set(avg_train_loss.keys()).union(set(avg_val_loss.keys()))

        with open(log_file_path, 'a') as log_file:
            log_file.write(f'Epoch {epoch} Summary:\n')
            for k in all_loss_keys:
                train_val_str = f'Train {k}: {avg_train_loss.get(k, 0):.4f} | Val {k}: {avg_val_loss.get(k, 0):.4f}'
                log_file.write(f'  {train_val_str}\n')
            log_file.write('\n')

        for k in all_loss_keys:
            train_val_str = f'Train {k}: {avg_train_loss.get(k, 0):.4f} | Val {k}: {avg_val_loss.get(k, 0):.4f}'
            print(f'  {train_val_str}')

        visualizer.plot_epoch_losses(epoch, {
            **{f'Train_{k}': v for k, v in avg_train_loss.items()},
            **{f'Val_{k}': v for k, v in avg_val_loss.items()}
        })

        model.update_learning_rate()

    print('\nTraining Finished.')
