from io import BytesIO
import torchvision
from osgeo import gdal
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import skimage.io
import cv2
import os
import csv
import random
import cv2
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import glob

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples to be sampled.
    Source code: https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

def has_black_pixels(tensor):
    # Sum along the channel dimension to get a 2D tensor [height, width]
    channel_sum = torch.sum(tensor, dim=0)

    # Check if any pixel has a sum of 0, indicating black
    black_pixels = (channel_sum.view(-1) == 0).any()

    return black_pixels


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', need_LR=False,
                    n_s2_images=-1, downsample_res=-1, output_size=512, max_tiles=-1, use_3d=False, specify_val=True):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.need_LR = need_LR
        self.split = split
        self.n_s2_images = n_s2_images
        self.downsample_res = downsample_res
        self.output_size = output_size
        self.max_tiles = max_tiles
        self.use_3d = use_3d
        
        self.s2_bands = ['tci']

        print("DATATYPE:", self.datatype)
        print("OUTPUT_SIZE:", self.output_size)
        print("DATAROOT:", dataroot)
        print("NUM LR IMAGES:", self.n_s2_images)

        ### WorldStrat case
        if datatype == 'worldstrat':
            self.all_bands = False
            self.use_3d = False

            # Hardcoded paths to data and splits
            self.splits_csv = dataroot + 'stratified_train_val_test_split.csv'
            self.lr_path = dataroot + 'lr_dataset/'
            self.hr_path = dataroot + 'hr_dataset/'

            # Read in the csv file containing splits and filter out non-relevant images for this split.
            # Build a list of [hr_path, [lr_paths]] lists. 
            self.datapoints = []
            with open(self.splits_csv, newline='') as csvfile:
                read = csv.reader(csvfile, delimiter=' ')
                for i,row in enumerate(read):
                    # Skip the row with columns.
                    if i == 0:
                        continue

                    row = row[0].split(',')
                    tile = row[1]
                    split = row[-1]
                    if split != self.split:
                        continue

                    # A few paths are missing even though specified in the split csv, so skip them.
                    if not os.path.exists((os.path.join(self.lr_path, tile, 'L1C', tile+'-'+str(1)+'-L1C_data.tiff'))):
                        continue

                    # HR image for the current datapoint. Still using rgb as ground truth (instead of pansharpened).
                    hr_img_path = os.path.join(self.hr_path, tile, tile+'_rgb.png')

                    # Each HR image has 16 corresponding LR images.
                    lrs = []
                    for img in range(1, int(self.n_s2_images)+1):
                        lr_img_path = os.path.join(self.lr_path, tile, 'L1C', tile+'-'+str(img)+'-L1C_data.tiff')
                        lrs.append(lr_img_path)

                    self.datapoints.append([hr_img_path, lrs])

                print("Loaded ", len(self.datapoints), " WorldStrat datapoints.")
                self.data_len = len(self.datapoints)
            return

        ### SEN2VENUS case
        elif datatype == 'sen2venus':
            self.output_size = 256
            data_root = '/data/piperw/data/sen2venus/' 
            hr_fps = glob.glob(data_root + '**/*_05m_b2b3b4b8.pt')

            # Filter filepaths based on if the split is train or validation.
            if self.split == 'train':
                hr_fps = [hr_fp for hr_fp in hr_fps if not ('JAM2018' in hr_fp or 'BENGA' in hr_fp or 'SO2' in hr_fp)]
            else:
                hr_fps = [hr_fp for hr_fp in hr_fps if ('JAM2018' in hr_fp or 'BENGA' in hr_fp or 'SO2' in hr_fp)]

            lr_fps = [hr.replace('05m', '10m') for hr in hr_fps]

            self.datapoints = []
            for i,hr_fp in enumerate(hr_fps):
                load_tensor = torch.load(hr_fp)
                num_patches = load_tensor.shape[0]
                self.datapoints.extend([[hr_fp, lr_fps[i], patch] for patch in range(num_patches)])

            print("Loaded ", len(self.datapoints), " SEN2Venus datapoints.")
            self.data_len = len(self.datapoints)
            return

        ### OLI2MSI case
        elif datatype == 'oli2msi':
            self.output_size = 160  # full-res output size is 480, but training on chunks
            self.data_root = '/data/piperw/data/OLI2MSI/'

            if self.split == 'train':
                hr_fps = glob.glob(self.data_root + 'train_hr/*.TIF')
                lr_fps = [hr_fp.replace('train_hr', 'train_lr') for hr_fp in hr_fps]
            else:
                hr_fps = hr_fps = glob.glob(self.data_root + 'test_hr/*.TIF')
                lr_fps = [hr_fp.replace('test_hr', 'test_lr') for hr_fp in hr_fps]

            self.datapoints = []
            for i,hr_fp in enumerate(hr_fps):
                self.datapoints.append([hr_fp, lr_fps[i]])

            print("Loaded ", len(self.datapoints), " OLI2MSI datapoints.")
            self.data_len = len(self.datapoints)
            return

        ### PROBA-V case
        elif datatype == 'probav':
            self.output_size = 120  # full-res output size is 384, but training on chunks
            self.data_root = '/data/piperw/data/PROBA-V/'

            hr_fps = glob.glob(self.data_root + 'train/NIR/*/HR.png')

            # Filter filepaths based on if the split is train or validation.
            if self.split == 'train':
                hr_fps = glob.glob(self.data_root + 'train/NIR/*/HR.png')
            else:
                hr_fps = glob.glob(self.data_root + 'train/NIR/val/*/HR.png')

            self.datapoints = []
            lr_fps = []
            for hr_fp in hr_fps:
                lrs = []
                for i in range(self.n_s2_images):
                    if i < 10:
                        lr = hr_fp.replace('HR', 'LR00' + str(i))
                    else:
                        lr = hr_fp.replace('HR', 'LR0' + str(i))
                    lrs.append(lr)
                self.datapoints.append([hr_fp, lrs])

            print("Loaded ", len(self.datapoints), " PROBA-V datapoints.")
            self.data_len = len(self.datapoints)
            return

        ### S2-NAIP case
        elif datatype == 's2naip':
            # Paths to Sentinel-2 and NAIP imagery.
            self.s2_path = dataroot + 'sentinel2/'
            self.naip_path = dataroot + 'naip/'
            if not (os.path.exists(self.s2_path) and os.path.exists(self.naip_path)):
                raise Exception("Please make sure the paths to the data directories are correct.")

            self.naip_chips = glob.glob(self.naip_path + '/**/*.png', recursive=True)

            # Reduce the training set down to a specified number of samples. If not specified, whole set is used.
            #if self.split == 'train':
            #    self.naip_chips = random.sample(self.naip_chips, 11000)

            print("self.naip chips:", len(self.naip_chips), " self.naip_path:", self.naip_path)

            self.datapoints = []
            for n in self.naip_chips:
                # Extract the X,Y chip from this NAIP image filepath.
                split_path = n.split('/')
                chip = split_path[-2]

                # Gather the filepaths to the Sentinel-2 bands specified in the config.
                s2_paths = [os.path.join(self.s2_path, chip, band + '.png') for band in self.s2_bands]

                self.datapoints.append([n, s2_paths])

            self.data_len = len(self.datapoints)
            print("Loaded ", len(self.datapoints), " S2NAIP datatpoints.")

    def get_tile_weight_sampler(self, tile_weights):
        weights = []
        for dp in self.datapoints:
            # Extract the NAIP chip from this datapoint's NAIP path.
            # With the chip, we can index into the tile_weights dict (naip_chip : weight)
            # and then weight this datapoint pair in self.datapoints based on that value.
            naip_path = dp[0]
            split = naip_path.split('/')[-1]
            chip = split[:-4]

            # If the chip isn't in the tile weights dict, then there weren't any OSM features
            # in that chip, so we can set the weight to be relatively low (ex. 1).
            if not chip in tile_weights:
                weights.append(1)
            else:
                weights.append(tile_weights[chip])

        print('using tile_weight_sampler, min={} max={} mean={}'.format(min(weights), max(weights), np.mean(weights)))
        return CustomWeightedRandomSampler(weights, len(self.datapoints))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        # Classifier-free guidance, X% of the time we want to replace S2 images with black images 
        # for "unconditional" generation during training. NOTE: currently hardcoded
        cfg = 5 # random.randint(0, 19)
        uncond = True if self.split == 'train' and cfg in [0,1,2,3] else False

        # S2NAIP
        if self.datatype == 's2naip':

            # A while loop and try/excepts to catch a few potential errors and continue if caught.
            counter = 0
            while True:
                index += counter  # increment the index based on what errors have been caught
                if index >= self.data_len:
                    index = 0

                datapoint = self.datapoints[index]
                naip_path, s2_paths = datapoint[0], datapoint[1]

                # Load the 128x128 NAIP chip in as a tensor of shape [channels, height, width].
                naip_chip = torchvision.io.read_image(naip_path)

                # Check for black pixels (almost certainly invalid) and skip if found.
                if has_black_pixels(naip_chip):
                    counter += 1
                    continue
                img_HR = naip_chip

                # Load the T*32x32xC S2 files for each band in as a tensor.
                # There are a few rare cases where loading the Sentinel-2 image fails, skip if found.
                try:
                    s2_tensor = None
                    for i,s2_path in enumerate(s2_paths):

                        # There are tiles where certain bands aren't available, use zero tensors in this case.
                        if not os.path.exists(s2_path):
                            img_size = (self.n_s2_images, 3, 32, 32) if 'tci' in s2_path else (self.n_s2_images, 1, 32, 32)
                            s2_img = torch.zeros(img_size, dtype=torch.uint8)
                        else:
                            s2_img = torchvision.io.read_image(s2_path)
                            s2_img = torch.reshape(s2_img, (-1, s2_img.shape[0], 32, 32))

                        # Upsample the low-res image to the size of the desired super-res size.
                        s2_img = F.interpolate(s2_img, (128, 128))

                        if i == 0:
                            s2_tensor = s2_img
                        else:
                            s2_tensor = torch.cat((s2_tensor, s2_img), dim=1)
                except:
                    counter += 1
                    continue

                # Skip the cases when there are not as many Sentinel-2 images as requested.
                if s2_tensor.shape[0] < self.n_s2_images:
                    counter += 1
                    continue

                # Iterate through the 32x32 tci chunks at each timestep, separating them into "good" (valid)
                # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
                tci_chunks = s2_tensor[:, :3, :, :]
                goods, bads = [], []
                for i,ts in enumerate(tci_chunks):
                    if has_black_pixels(ts):
                        bads.append(i)
                    else:
                        goods.append(i)

                # Pick self.n_s2_images random indices of S2 images to use. Skip ones that are partially black.
                if len(goods) >= self.n_s2_images:
                    rand_indices = random.sample(goods, self.n_s2_images)
                else:
                    need = self.n_s2_images - len(goods)
                    rand_indices = goods + random.sample(bads, need)
                rand_indices_tensor = torch.as_tensor(rand_indices)

                # Extract the self.n_s2_images from the first dimension.
                img_S2 = s2_tensor[rand_indices_tensor]

                # If using a model that expects 5 dimensions, we will not reshape to 4 dimensions.
                if not self.use_3d:
                    img_S2 = torch.reshape(img_S2, (-1, 128, 128))

                break

            img_S2 = img_S2.float() / 255
            img_HR = img_HR.float() / 255

            # Classifier-free guidance step, replace S2 images with all black images.
            if uncond:
                img_S2 = torch.zeros_like(img_S2)

            return {'HR': img_HR, 'SR': img_S2, 'Index': index}

        elif self.datatype == 'worldstrat':
            hr_path, lr_paths = self.datapoints[index]

            hr_im = torchvision.io.read_image(hr_path)[0:3, :, :]  # remove alpha channel
            img_HR = F.interpolate(hr_im, 640) # resize the HR image to match the SR image

            # Load each of the LR images with gdal, since they're tifs.
            lr_ims = []
            for lr_path in lr_paths:
                raster = gdal.Open(lr_path)
                array = raster.ReadAsArray()
                lr_im = array.transpose(1, 2, 0)[:, :, 1:4]  # only using RGB bands (bands 2,3,4)
                lr_im = torch.permute(torch.tensor(cv2.resize(lr_im, (160,160))), (2, 1, 0))
                lr_ims.append(lr_im)

            # Resize each Sentinel-2 image to the same spatial dimension. Then stack along first dimension.
            img_LR = torch.stack(lr_ims, dim=0)

            # Find a random 160x160 HR chunk, to create more, smaller training samples.
            hr_start_x = random.randint(0, 640-160)
            hr_start_y = random.randint(0, 640-160)
            lr_start_x = hr_start_x // 4
            lr_start_y = hr_start_y // 4

            img_HR = img_HR[:, hr_start_x:hr_start_x+160, hr_start_y:hr_start_y+160]
            img_LR = img_LR[:, :, lr_start_x:lr_start_x+40, lr_start_y:lr_start_y+40]

            img_LR = F.interpolate(img_LR, (160,160))
            img_LR = torch.reshape(img_LR, (-1, 160, 160))

            img_HR = img_HR.float() / 255
            img_LR = img_LR.float()

            return {'HR': img_HR, 'SR': img_LR, 'Index': index}

        elif self.datatype == 'sen2venus':
            hr_path, lr_path, patch_num = self.datapoints[index]

            hr_tensor = torch.load(hr_path)[patch_num, :3, :, :].float()
            lr_tensor = torch.load(lr_path)[patch_num, :3, :, :].float()
            lr_tensor = F.interpolate(lr_tensor.unsqueeze(0), (256, 256)).squeeze(0)

            if self.use_3d:
                lr_tensor = lr_tensor.unsqueeze(0)

            img_HR = hr_tensor
            img_LR = lr_tensor
            return {'HR': img_HR, 'SR': img_LR, 'Index': index}
		
        elif self.datatype == 'oli2msi':
            hr_path, lr_path = self.datapoints[index]

            # Load the 480x840 high-res image.
            hr_ds = gdal.Open(hr_path)
            hr_arr = np.array(hr_ds.ReadAsArray())
            hr_tensor = torch.tensor(hr_arr)

            # Load the 160x160 low-res image.
            lr_ds = gdal.Open(lr_path)
            lr_arr = np.array(lr_ds.ReadAsArray())
            lr_tensor = torch.tensor(lr_arr)

            # Find a random 120x120 HR chunk, to create more, smaller training samples.
            # Reshape the 120x120 chunk up to 160x160 so our 4x training module will work.
            hr_start_x = random.randint(0, 480-120)
            hr_start_y = random.randint(0, 480-120)
            lr_start_x = int(hr_start_x // 3)
            lr_start_y = int(hr_start_y // 3)

            hr_tensor = hr_tensor[:, hr_start_x:hr_start_x+120, hr_start_y:hr_start_y+120]
            lr_tensor = lr_tensor[:, lr_start_x:lr_start_x+40, lr_start_y:lr_start_y+40]

            hr_tensor = F.interpolate(hr_tensor.unsqueeze(0), (160,160)).squeeze(0)
            lr_tensor = F.interpolate(lr_tensor.unsqueeze(0), (160,160)).squeeze(0)  # upsample to desired output size

            img_HR = hr_tensor.float()
            img_LR = lr_tensor.float()
            return {'HR': img_HR, 'SR': img_LR, 'Index': index}

        elif self.datatype == 'probav':
            hr_path, lr_paths  = self.datapoints[index]

            hr_im = cv2.imread(hr_path)

            # Take a random 120x120 HR chunk; Reshape the 120x120 chunk up to 160x160 so our 4x training module will work.
            rand_start_x = random.randint(0, 263)
            rand_start_y = random.randint(0, 263)
            hr_im = hr_im[rand_start_x:rand_start_x+120, rand_start_y:rand_start_y+120, :]
            hr_tensor = torch.permute(torch.tensor(hr_im), (2, 0, 1))
            img_HR = F.interpolate(hr_tensor.unsqueeze(0), (160,160)).squeeze(0)

            lr_start_x = int(rand_start_x // 3)
            lr_start_y = int(rand_start_y // 3)

            lr_ims = []
            for lr_path in lr_paths:
                lr_im = cv2.imread(lr_path)
                lr_im = lr_im[lr_start_x:lr_start_x+40, lr_start_y:lr_start_y+40, :]
                lr_im = cv2.resize(lr_im, (160,160))
                lr_tensor = torch.permute(torch.tensor(lr_im), (2, 0, 1))
                lr_ims.append(lr_tensor)
            img_LR = torch.cat(lr_ims)

            img_HR = img_HR.float() / 255
            img_LR = img_LR.float() / 255

            return {'HR': img_HR, 'SR': img_LR, 'Index': index}
