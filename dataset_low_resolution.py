import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from monai.transforms import ScaleIntensity
# from monai.transforms import SpatialPadd


class PelvisDataset(Dataset):
    def __init__(self, root_dir, input_dim=(224, 224, 224), nlabels=4, transform=None):
        super(PelvisDataset, self).__init__()

        self.root_dir = root_dir
        self.input_dim = input_dim
        self.transform = transform
        self.nlabels = nlabels

        # define the path for images and their labels
        self.images_path = self.root_dir + '/images/'
        self.labels_path = self.root_dir + '/landmarks/'

        # sort the list of images and labels so that they match
        self.images = sorted(os.listdir(self.images_path))
        self.labels = sorted(os.listdir(self.labels_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Select the item of interest
        img_name = self.images[idx]
        label_name = self.labels[idx]
        # print(img_name)

        # read the images and labels and convert them to numpy arrays
        image = sitk.ReadImage(self.images_path + img_name)
        label = sitk.ReadImage(self.labels_path + label_name)

        image = sitk.SmoothingRecursiveGaussian(image, sigma=[1, 1, 1])
        # label = sitk.SmoothingRecursiveGaussian(label, sigma=[3, 3, 3])

        
        # Transform them if specified
        if self.transform:
            image, label = self.transform((image, label))

        
        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)
        
        # image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
        # axes[1].imshow(image_array[22,:,:],cmap="gray")
        # axes[1].set_title("before norm")
        # axes[0].imshow(image[22,:,:],cmap="gray")
        # axes[0].set_title("afternorm")
        
        # print('image size', image.shape)
        # print("label size", label.shape)
        # Apply the transform to your image
        
        # label_t = np.transpose(label, (2,1,0) )
        # axes[1].imshow(label_t[22,:,:],cmap="gray")
        # axes[1].set_title("label")
        # axes[0].imshow(image[22,:,:],cmap="gray")
        # axes[0].set_title("image")
        
        label_img = np.copy(label)
        

        # Establecer todos los valores distintos de 0 a 1
        
        label_img[abs(label_img) > 1e-3] = 1
        label_img[abs(label_img) <= 1e-3] = 0
        # plt.imshow(label_img[22,:,:])
        # plt.show()
        # image = padding_layer(image)
        # label = padding_layer(label_t)
        # print('label resize', label_t.shape)
        # Normalize the image
        # image = (image - np.mean(image)) / np.std(image)

        # Convert label to one-hot encoding
        # label = label_t.astype(int)
        # plt.imshow(label[22,:,:],cmap ="gray")
        # plt.show()
        # label[label >= self.nlabels] = 0
        # label_one_hot = torch.nn.functional.one_hot(torch.from_numpy(label),
                                                 # self.nlabels + 1).float()
        label_one_hot = torch.from_numpy(label_img).float().unsqueeze(0)
        
        # change the order so that it is CxWxHxD instead of WxHxDxC
        # label_tensor = label_one_hot.permute(3, 0, 1, 2)
        label_tensor = label_one_hot[0:1,:,:,:]
        
        # Convert image to PyTorch tensor so that dimensions are CxWxHxD
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        
        # padding_layer = nn.ConstantPad3d(padding=(31, 31, 54, 54, 75, 76), value=0)
        padding_layer = nn.ConstantPad3d(padding=(7, 7, 6, 6, 4, 3), value=0)
        
        image_tensor = padding_layer(image_tensor)
        label_tensor = padding_layer(label_tensor)

        # print("label tensor", label_tensor.shape)
        # print("image tensor", image_tensor.shape)

        return image_tensor, label_tensor, img_name, label_name

# import os
# import torch
# from torch.utils.data import Dataset
# import SimpleITK as sitk
# import numpy as np
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from monai.transforms import ScaleIntensity
# from monai.data import DataLoader, CacheDataset, ImageDataset, Dataset, image_reader
# # from monai.transforms import SpatialPadd


# class PelvisDataset(Dataset):
#     def __init__(self, root_dir, input_dim=(224, 224, 224), nlabels=4, transform=None):
#         super(PelvisDataset).__init__()

#         self.root_dir = root_dir
#         self.input_dim = input_dim
#         self.transform = transform
#         self.nlabels = nlabels

#         # define the path for images and their labels
#         self.images_path = self.root_dir + '/images/'
#         self.labels_path = self.root_dir + '/landmarks/'

#         # sort the list of images and labels so that they match
#         self.images = sorted(os.listdir(self.images_path))
#         self.labels = sorted(os.listdir(self.labels_path))

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         # Select the item of interest
#         img_name = self.images[idx]
#         label_name = self.labels[idx]
#         # print(img_name)

#         # read the images and labels and convert them to numpy arrays
#         image = sitk.ReadImage(self.images_path + img_name)
#         label = sitk.ReadImage(self.labels_path + label_name)

        
#         # Transform them if specified
#         if self.transform:
#             image, label = self.transform((image, label))

        
#         image = sitk.GetArrayFromImage(image)
#         label = sitk.GetArrayFromImage(label)
        
#         # image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        
#         # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
#         # axes[1].imshow(image_array[22,:,:],cmap="gray")
#         # axes[1].set_title("before norm")
#         # axes[0].imshow(image[22,:,:],cmap="gray")
#         # axes[0].set_title("afternorm")
        
#         # print('image size', image.shape)
#         # print("label size", label.shape)
#         # Apply the transform to your image
        
#         # label_t = np.transpose(label, (2,1,0) )
#         label_t = label
#         # axes[1].imshow(label_t[22,:,:],cmap="gray")
#         # axes[1].set_title("label")
#         # axes[0].imshow(image[22,:,:],cmap="gray")
#         # axes[0].set_title("image")
        
#         label_img = np.copy(label_t)
        

#         # Establecer todos los valores distintos de 0 a 1
        
#         label_img[abs(label_img) > 1e-3] = 1
#         label_img[abs(label_img) <= 1e-3] = 0

#         label_one_hot = torch.from_numpy(label_img).float().unsqueeze(0)
        

#         label_tensor = label_one_hot[0:1,:,:,:]
        
#         # Convert image to PyTorch tensor so that dimensions are CxWxHxD
#         image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        
#         desired_size = [256,256,256]
#         im_size=image.shape
#         # padding_layer = nn.ConstantPad3d(padding=( (desired_size[0]-im_size[0]//2), (desired_size[0]-im_size[0]//2), (desired_size[1]-im_size[1]//2), (desired_size[1]-im_size[1]//2), (desired_size[2]-im_size[2]//2), (desired_size[2]-im_size[2]//2)), value=0)
        
#         padding_layer = nn.ConstantPad3d(padding=( 3, 3, 5, 5, 7, 7), value=0)
#         image_tensor = padding_layer(image_tensor)
#         label_tensor = padding_layer(label_tensor)
#         image_tensor = image_tensor.permute(0,3,2,1)
#         label_tensor = label_tensor.permute(0,3,2,1)
#         print('final image tensor, ', image_tensor.shape)
#         print('final label tensor, ', label_tensor.shape)

#         return image_tensor, label_tensor


#     # def plotitem(self, idx, slc=None):
#     #     # Select the item of interest
#     #     img_name = self.images[idx]
#     #     label_name = self.labels[idx]

#     #     # Read the images and labels
#     #     image = sitk.ReadImage(self.images_path + img_name)
#     #     image = np.array(sitk.GetArrayFromImage(image))

#     #     label = sitk.ReadImage(self.labels_path + label_name)
#     #     label = np.array(sitk.GetArrayFromImage(label))

#     #     # Transform them if specified
#     #     if self.transform:
#     #         sample = (image, label)
#     #         sample = self.transform(sample)
#     #         image, label = sample

#     #     # Select the middle slice if not specified
#     #     if slc is None:
#     #         slc = image.shape[2] // 2

#     #     image = sitk.GetImageFromArray(image)
#     #     label = sitk.GetImageFromArray(label)

#     #     fig, axs = plt.subplots(2, 3, figsize=(10, 8))

#     #     axs[0, 0].imshow(sitk.GetArrayViewFromImage(image)[:, :, slc], cmap=plt.cm.Greys_r)
#     #     axs[0, 0].set_title(f'Sagittal plane image {img_name.strip(".nii.gz")}')

#     #     axs[0, 1].imshow(sitk.GetArrayViewFromImage(image)[:, slc, :], cmap=plt.cm.Greys_r)
#     #     axs[0, 1].set_title(f'Coronal Plane {img_name.strip(".nii.gz")}')

#     #     axs[0, 2].imshow(sitk.GetArrayViewFromImage(image)[slc, :, :], cmap=plt.cm.Greys_r)
#     #     axs[0, 2].set_title(f'Axial plane {img_name.strip(".nii.gz")}')

#     #     axs[1, 0].imshow(sitk.GetArrayViewFromImage(label)[:, :, slc], cmap=plt.cm.Greys_r)
#     #     axs[1, 0].set_title(f'Sagittal plane label {label_name.strip(".nii.gz")}')

#     #     axs[1, 1].imshow(sitk.GetArrayViewFromImage(label)[:, slc, :], cmap=plt.cm.Greys_r)
#     #     axs[1, 1].set_title(f'Coronal Plane {label_name.strip(".nii.gz")}')

#     #     axs[1, 2].imshow(sitk.GetArrayViewFromImage(label)[slc, :, :], cmap=plt.cm.Greys_r)
#     #     axs[1, 2].set_title(f'Axial plane {label_name.strip(".nii.gz")}')

#     #     plt.tight_layout()
#     #     plt.show()
