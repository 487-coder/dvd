import os
import time
import torch
import torchvision.utils as tutils
from new_utils import batch_psnr
from new_dvdnet import denoise_seq_dvdnet
from pathlib import Path
from models import Spatial_block,Temporal_block
import torch.nn as nn


def resume_training(config, model, optimizer):
    """Resume training from checkpoint if available."""
    if config.get('resume_training', False):
        log_dir = Path(config['log_dir'])
        resume_path = log_dir / 'ckpt_latest.pth'

        if resume_path.is_file():
            checkpoint = torch.load(resume_path)
            print("> Resuming previous training...")

            # 分别加载 spatial 和 temporal 模型
            spatial_model = model.spatial_model.module if isinstance(model.spatial_model, nn.DataParallel) else model.spatial_model
            temporal_model = model.temporal_model.module if isinstance(model.temporal_model, nn.DataParallel) else model.temporal_model

            spatial_model.load_state_dict(checkpoint['spatial_model'])
            temporal_model.load_state_dict(checkpoint['temporal_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 恢复训练参数
            training_params = checkpoint.get('training_params', {'start_epoch': 0, 'step': 0, 'no_orthog': False})
            start_epoch = checkpoint.get('epoch', 0)

            # 覆盖旧配置以支持新超参数设置
            weight = checkpoint.get('args', {})
            weight.update({
                'epochs': config['epochs'],
                'milestone': config['milestone'],
                'lr': config['lr'],
                'resume_training': False
            })

            print(f"=> Loaded checkpoint '{resume_path}' (epoch {start_epoch})")
            print("==> Optimizer param_groups:")
            print(f"{optimizer.param_groups}")

            print("==> Training params:")
            for k, v in training_params.items():
                print(f"\t{k}: {v}")

            print("==> Args:")
            for k, v in weight.items():
                print(f"\t{k}: {v}")

            return start_epoch, training_params
        else:
            raise FileNotFoundError(f"Couldn't resume training, checkpoint not found: {resume_path}")
    else:
        # Start from scratch
        return 0, {'start_epoch': 0, 'step': 0, 'no_orthog': False}


def lr_scheduler(config, epoch):
    reset_orthogonal = False
    if epoch > config['milestone'][1]:
        current_lr = config['lr'] / 1000.
        reset_orthogonal = True
    elif epoch > config['milestone'][0]:
        current_lr = config['lr'] / 10.
    else:
        current_lr = config['lr']
    return current_lr, reset_orthogonal


def log_train_psnr(result, source, loss, writer, epoch, idx, num_minibatches, training_params):
    writer.add_scalar('loss', loss.item(), training_params['step'])
    psnr_train = batch_psnr(result, source, 1.0)
    print(f"[epoch {epoch + 1}][{idx + 1}/{num_minibatches}] loss: {loss.item():.4f} PSNR_train: {psnr_train:.4f}")
    global_step = epoch * num_minibatches + idx
    writer.add_scalar("Loss/train", loss.item(), global_step)
    writer.add_scalar("PSNR/train", psnr_train, global_step)


'''
def save_model_checkpoint(model, config, optimizer, train_pars, epoch):
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), log_dir / 'net.pth')

    save_dict = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'training_params': train_pars,
        'args': config
    }

    torch.save(save_dict, log_dir / 'ckpt.pth')

    if epoch % config['save_every_epochs'] == 0:
        epoch_ckpt_path = log_dir / f'ckpt_e{epoch + 1}.pth'
        torch.save(save_dict, epoch_ckpt_path)

    del save_dict
'''
def save_model_checkpoint(model, config, optimizer, train_pars, epoch):
    """
    Save model checkpoints for spatial and temporal blocks, along with optimizer and training metadata.

    Args:
        model: a nn.Module containing spatial_model and temporal_model as attributes.
        config: dict containing at least 'log_dir' and 'save_every_epochs'.
        optimizer: optimizer object.
        train_pars: dictionary of training parameters.
        epoch: current training epoch (int).
    """
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Handle potential DataParallel wrappers
    spatial_model = model.spatial_model.module if isinstance(model.spatial_model, nn.DataParallel) else model.spatial_model
    temporal_model = model.temporal_model.module if isinstance(model.temporal_model, nn.DataParallel) else model.temporal_model

    # Save model states
    torch.save(spatial_model.state_dict(), log_dir / f'spatial_e{epoch+1}.pth')
    torch.save(temporal_model.state_dict(), log_dir / f'temporal_e{epoch+1}.pth')

    # Save training metadata
    save_dict = {
        'epoch': epoch + 1,
        'optimizer': optimizer.state_dict(),
        'training_params': train_pars,
        'args': config
    }
    torch.save(save_dict, log_dir / 'ckpt_latest.pth')

    # Save periodic checkpoint
    if (epoch + 1) % config.get('save_every_epochs', 1) == 0:
        torch.save(save_dict, log_dir / f'ckpt_e{epoch+1}.pth')

    print(f"Checkpoint saved at epoch {epoch+1} in {log_dir}")



from pathlib import Path
import time
import torch
import torchvision.utils as tutils
from skimage.metrics import peak_signal_noise_ratio


def validate_and_log(Spatial_block, Temporal_block,  dataset_val, valnoisestd, temp_psz, writer,
                     epoch, lr, logger, trainimg):
    """Run validation and log PSNR + sample images."""

    t1 = time.time()
    psnr_val = 0
    for seq_val in dataset_val:
        # Add Gaussian noise
        noise = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
        seqn_val = (seq_val + noise).cuda()
        sigma_noise = torch.tensor([valnoisestd], device='cuda')
        out_val = denoise_seq_dvdnet(
            seq=seqn_val,
            noise_std=sigma_noise,
            temporal_patch= temp_psz,
            spatial_model= Spatial_block,
            temporal_model= Temporal_block
        )
        psnr_val += batch_psnr(out_val.cpu(), seq_val.squeeze_(), data_range=1.0)

    psnr_val /= len(dataset_val)
    t2 = time.time()

    print(f"\n[epoch {epoch + 1}] PSNR_val: {psnr_val:.4f}, on {t2 - t1:.2f} sec")
    writer.add_scalar('PSNR on validation data', psnr_val, epoch)
    writer.add_scalar('Learning rate', lr, epoch)

    try:
        idx = 0
        if epoch == 0:
            # Show training images
            _, _, Ht, Wt = trainimg.shape
            grid_train = tutils.make_grid(trainimg.view(-1, 3, Ht, Wt),
                                          nrow=8, normalize=True, scale_each=True)
            writer.add_image('Training patches', grid_train, epoch)

            # Show clean and noisy validation images
            clean_img = tutils.make_grid(seq_val.data[idx].clamp(0., 1.),
                                         nrow=2, normalize=False, scale_each=False)
            noisy_img = tutils.make_grid(seqn_val.data[idx].clamp(0., 1.),
                                         nrow=2, normalize=False, scale_each=False)
            writer.add_image(f'Clean validation image {idx}', clean_img, epoch)
            writer.add_image(f'Noisy validation image {idx}', noisy_img, epoch)

        # Reconstructed output
        recon_img = tutils.make_grid(out_val.data[idx].clamp(0., 1.),
                                     nrow=2, normalize=False, scale_each=False)
        writer.add_image(f'Reconstructed validation image {idx}', recon_img, epoch)

    except Exception as e:
        logger.error(f"validate_and_log(): Couldn't log results, {e}")
