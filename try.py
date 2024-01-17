import os
import monai
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.data import DataLoader, CacheDataset, ImageDataset, Dataset, image_reader
from monai.transforms import Compose, ScaleIntensity, ToTensor,LoadImaged, ScaleIntensityd, ToTensord,EnsureChannelFirstd
import SimpleITK as sitk
from monai.data.image_reader import ITKReader
import dataset
from dataset import PelvisDataset

root_dir ="/home/paulagmtz/TFM_PAULA_24/DATA_TFM/train_files"
input_dim = (194,148,105)
n_labels= 2
Dataset = PelvisDataset(root_dir,input_dim, n_labels, transform=None)

train_dataloader = DataLoader(Dataset, batch_size =2)