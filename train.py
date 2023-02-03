# Ampatishan
# 11-22-2022
# https://github.com/milesial/Pytorch-UNet

import warnings
import wandb
# import torchio

import torch.nn.functional
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from monai.losses import DiceCELoss, Dice

import src.unet_model as net
from utils.dataset import paip_dataset

warnings.filterwarnings("ignore")

dice = Dice()
DiceCELoss = DiceCELoss()


def train(config, model=net):
    with wandb.init(project="Paip_challenge", config=config):
        config = wandb.config
        transform = transforms.Compose([

            transforms.ToTensor(),
            # transforms.Resize(64)
        ])
        train_dataset = paip_dataset(is_colon=False,
                                     image_path=config.tr_img_path,
                                     mask_path=config.tr_masks_path,
                                     transforms=transform,
                                     target_transfrom=transform)

        val_dataset = paip_dataset(is_colon=False,
                                   image_path=config.val_img_path,
                                   mask_path=config.val_masks_path,
                                   transforms=transform,
                                   target_transfrom=transform)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=config.batch_size)

        model = model.UNet(3, 2)
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                  momentum=config.momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        global_step = 0
        epoch_loss = 0

        epochs = config.epochs
        device = 'cuda'
        model.to(device)

        wandb.watch(model, DiceCELoss, log="all", log_freq=10)
        for epoch in range(1, epochs + 1):
            model.train()

            with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    tr_images = batch[0]
                    tr_gt = batch[1]

                    assert tr_images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {tr_images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    tr_images = tr_images.to(device=device, dtype=torch.float32)
                    tr_gt = tr_gt.to(device=device, dtype=torch.long)

                    with torch.cuda.amp.autocast(enabled=False):
                        tr_pred = model(tr_images)
                        loss = DiceCELoss(tr_pred, tr_gt)

                    optimizer.zero_grad(set_to_none=True)

                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(tr_images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    wandb.log({"epoch": epoch, "loss": loss.item(), "global_step": global_step})
                print('train loss: ' + str(epoch_loss / global_step))

                model.eval()
                num_val_batches = len(val_loader)
                dice_coeff = 0
                val_loss = 0

                # iterate over the validation set
                for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch',
                                  leave=False):
                    val_image, val_gt = batch[0], batch[1]
                    # move images and labels to correct device and type
                    val_image = val_image.to(device=device, dtype=torch.float32)
                    val_gt = val_gt.to(device=device, dtype=torch.long)

                    with torch.no_grad():
                        # predict the mask
                        val_pred = model(val_image)
                        val_loss += DiceCELoss(val_pred, val_gt).item()
                        wandb.log({'val_loss': DiceCELoss(val_pred, val_gt).item()})
                        dice_coeff += 1 - dice(val_pred, val_gt)
                print('val_loss : ' + str(val_loss / num_val_batches))
                print('dice_score : ' + str(dice_coeff / num_val_batches))

                wandb.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation dice': dice_coeff,
                    # 'images': wandb.Image(images[0].cpu()),
                    # 'masks': {
                    #     'true': wandb.Image(true_masks[0].float().cpu()),
                    #     'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                    # },
                    'step': global_step,
                    'epoch': epoch
                })
            #
            # if save_checkpoint:
            #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            #     torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            #     logging.info(f'Checkpoint {epoch} saved!')
