import torch
import random
from PIL import Image
import skimage.io
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
from torchvision.transforms import functional as trans_fn
from osgeo import gdal
import torch.nn.functional as F

import data.util as Util
from metrics import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_txt', type=str, help="Path to a txt file with a list of naip filepaths.")
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('-save_path', default='outputs')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    save_path = args.save_path

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # If not resuming from last checkpoint and just trying to load in weights, it should default to here.
    current_epoch = 2000
    current_step = 1000
    if (opt['path']['resume_gen_state'] and opt['path']['resume_opt_state']) or opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    #for phase, dataset_opt in opt['datasets'].items():
    #    if phase == 'val':
    #        val_set = Data.create_dataset(dataset_opt, phase, output_size=opt['datasets']['output_size'])
    #        val_loader = Data.create_dataloader(
    #            val_set, dataset_opt, phase)
    #logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    avg_psnr, avg_ssim = [], []

    device = torch.device('cuda')

    data_txt = args.data_txt
    if 'oli2msi' in data_txt:
        datatype = 'oli2msi'
        base_path = '/data/piperw/data/OLI2MSI/'
        save_path = '/data/piperw/cvpr_outputs/oli2msi/'
    elif 'sen2venus' in data_txt:
        sen2venus_counter = 0
        datatype = 'sen2venus'
        base_path = '/data/piperw/data/sen2venus/'
        save_path = '/data/piperw/cvpr_outputs/sen2venus/'
    else:
        datatype = 'naip-s2'
        base_path = '/data/piperw/data/val_set/'
    print("Datatype:", datatype)

    n_s2_images = 8
    txt = open(data_txt)
    fps = txt.readlines()
    for idx,png in enumerate(fps):
        print("Processing....", idx)

        png = png.replace('\n', '')

        if datatype == 'naip-s2':

            # Want to save the super-resolved imagery in the same filepath structure 
            # as the Sentinel-2 imagery, but in a different directory specified by args.save_path
            # for easy comparison.
            file_info = png.split('/')
            chip = file_info[-1][:-4]
            save_dir = os.path.join(save_path, chip)
            os.makedirs(save_dir, exist_ok=True)

            # Load and format NAIP image as diffusion code expects.
            naip_chip = skimage.io.imread(png)
            #skimage.io.imsave(save_dir + '/naip.png', naip_im)

            chip = chip.split('_')
            tile = int(chip[0]) // 16, int(chip[1]) // 16

            s2_left_corner = tile[0] * 16, tile[1] * 16
            diffs = int(chip[0]) - s2_left_corner[0], int(chip[1]) - s2_left_corner[1]

            # Load and format S2 images as diffusion code expects.
            s2_path = base_path + 's2_condensed/' + str(tile[0])+'_'+str(tile[1]) + '/' + str(diffs[1])+'_'+str(diffs[0]) + '.png'
            s2_images = skimage.io.imread(s2_path)
            s2_chunks = np.reshape(s2_images, (-1, 32, 32, 3))

            goods, bads = [], []
            for i,ts in enumerate(s2_chunks):
                if [0, 0, 0] in ts:
                    bads.append(i)
                else:
                    goods.append(i)
            if len(goods) >= n_s2_images:
                rand_indices = random.sample(goods, n_s2_images)
            else:
                need = n_s2_images - len(goods)
                rand_indices = goods + random.sample(bads, need)

            s2_chunks = [s2_chunks[i] for i in rand_indices]
            s2_chunks = np.array(s2_chunks)
            up_s2_chunk = torch.permute(torch.from_numpy(s2_chunks), (0, 3, 1, 2))
            up_s2_chunk = trans_fn.resize(up_s2_chunk, (128,128), Image.BICUBIC, antialias=True)
            s2_chunks = torch.permute(up_s2_chunk, (0, 2, 3, 1)).numpy()
            [s2_chunks, img_HR] = Util.transform_augment(
                            [s2_chunks, naip_chip], split='val',  min_max=(-1, 1), multi_s2=True)
            img_SR = torch.cat(s2_chunks).unsqueeze(0)
            img_HR = img_HR.unsqueeze(0)

        elif datatype == 'oli2msi':
            save_dir = os.path.join(save_path, str(idx))
            os.makedirs(save_dir, exist_ok=True)

            lr_path = base_path + png
            hr_path = lr_path.replace('test_lr', 'test_hr')

            hr_ds = gdal.Open(hr_path)
            hr_arr = np.array(hr_ds.ReadAsArray())
            hr_arr = np.transpose(cv2.resize(np.transpose(hr_arr, (1, 2, 0)), (320, 320)), (2, 0, 1))
            hr_tensor = torch.tensor(hr_arr).float()

            # Uncomment if you don't want to save high-res image.
            hr_save = (np.transpose(hr_arr, (1, 2, 0)) * 255).astype(np.uint8)
            cv2.imwrite(save_dir + '/hr.png', hr_save)

            lr_ds = gdal.Open(lr_path)
            lr_arr = np.array(lr_ds.ReadAsArray())
            lr_tensor = torch.tensor(lr_arr).float()
            lr_tensor = F.interpolate(lr_tensor.unsqueeze(0), (320, 320))

            img_HR = hr_tensor
            img_SR = lr_tensor

        elif datatype == 'sen2venus':
            # Only grabbing the RGB S2 image
            if not '10m_b2b3b4b8' in png:
                continue

            lr_path = base_path + png
            hr_path = lr_path.replace('10m', '05m')

            hr_tensor = torch.load(hr_path)[:, :3, :, :].float().to(device)
            lr_tensor = torch.load(lr_path)[:, :3, :, :].float().to(device)

            for patch in range(hr_tensor.shape[0]):

                save_dir = os.path.join(save_path, str(sen2venus_counter))
                os.makedirs(save_dir, exist_ok=True)
                sen2venus_counter += 1

                lr_patch = lr_tensor[patch, :, :, :].unsqueeze(0)
                hr_patch = lr_tensor[patch, :, :, :]

                img_HR = hr_patch
                img_SR = lr_patch

                val_data = {'HR': img_HR, 'SR': img_SR, 'Index': torch.tensor(idx)}

                diffusion.feed_data(val_data)
                diffusion.test(continous=False)
                visuals = diffusion.get_current_visuals(need_LR=False)

                hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                sr_img = Metrics.tensor2img(visuals['SR'])
                Metrics.save_img(sr_img, save_dir + '/sr3.png')

            continue

        val_data = {'HR': img_HR, 'SR': img_SR, 'Index': torch.tensor(idx)}

        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img = Metrics.tensor2img(visuals['SR'])
        #sr_img = Metrics.tensor2img(visuals['SR'][:, -1, :, :, :])
        Metrics.save_img(sr_img, save_dir + '/sr3.png')

        eval_psnr = calculate_psnr(hr_img, sr_img, 0)
        eval_ssim = calculate_ssim(hr_img, sr_img, 0)
        print(eval_psnr, eval_ssim)

        avg_psnr.append(eval_psnr)
        avg_ssim.append(eval_ssim)

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    avg_psnr = sum(avg_psnr) / len(avg_psnr)
    avg_ssim = sum(avg_ssim) / len(avg_ssim)

    print("Avg PSNR:", avg_psnr)
    print("Avg SSIM:", avg_ssim)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
