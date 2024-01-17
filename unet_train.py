import os
import monai
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.data import DataLoader, CacheDataset, ImageDataset, Dataset, image_reader
from monai.data.utils import pad_list_data_collate
from monai.transforms import Compose, ScaleIntensity, ToTensor,LoadImaged, ScaleIntensityd, ToTensord,EnsureChannelFirstd
import SimpleITK as sitk
from monai.data.image_reader import ITKReader
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, FocalLoss, GeneralizedDiceLoss


from monai.metrics import MSEMetric
import dataset

from dataset import PelvisDataset
import numpy as np
import sklearn
from monai.metrics import compute_hausdorff_distance
from UNETR import UNETR
import ipywidgets as widgets
from IPython.display import display
import time
import torch.nn.functional as F






# train
root_dir ="/home/paulagmtz/TFM_PAULA_24/DATA_TFM/train_files"
input_dim = (194,148,105)
n_labels= 6
train_dataset = PelvisDataset(root_dir,input_dim, n_labels, transform=None)

train_dataloader = DataLoader(train_dataset, collate_fn=pad_list_data_collate,batch_size =2)

#val
# root_dir ="/home/paulagmtz/TFM_PAULA_24/DATA_TFM/val_files"
# input_dim = (194,148,105)
# n_labels= 2
# val_dataset = PelvisDataset(root_dir,input_dim, n_labels, transform=None)

# val_dataloader = DataLoader(val_dataset, batch_size =2)

# #test
# root_dir ="/home/paulagmtz/TFM_PAULA_24/DATA_TFM/test_files"
# input_dim = (194,148,105)
# n_labels= 2
# test_dataset = PelvisDataset(root_dir,input_dim, n_labels, transform=None)

# test_dataloader = DataLoader(test_dataset, batch_size =2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims=3 ,
    in_channels=1,
    out_channels=7,
    channels=(64,128,256,512), 
    strides=(2,2,2),
    # num_res_units=2,
    # norm=Norm.BATCH,    
).to(device)


# Unetr model



# model = UNETR(in_channels=1, out_channels=1, img_size=(112, 160, 208), feature_size=32, norm_name='batch')
# model.to(device)

# loss and optimizer

dice_loss_function = DiceLoss(sigmoid=True, squared_pred=True) 
diceCE_loss_function = DiceCELoss(sigmoid=True)
dicefocall_loss_function = DiceFocalLoss(sigmoid=True)
generalizedDice_loss_function = GeneralizedDiceLoss(sigmoid=True)


optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Función para mostrar imágenes de inputs, targets y outputs
def plot_results(inputs, targets, outputs):
    depth_slice = inputs[:, :, :, :, :]
    depth_slice_target = targets[:, :, :, :, :]
    depth_slice_outputs = outputs[:, :, :, :, :]

    def plot_image(i):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        image_array = depth_slice[0, :, i, :, :].cpu().detach().numpy()
        image_array_target = depth_slice_target[0, :, i, :, :].cpu().detach().numpy()
        image_array_outputs = depth_slice_outputs[0, :, i, :, :].cpu().detach().numpy()

        axes[0].imshow(image_array[0, :, :], cmap="gray")
        axes[0].set_title("inputs")
        axes[0].axis('off')

        axes[1].imshow(image_array_target[0, :, :], cmap="gray")
        axes[1].set_title("target")
        axes[1].axis('off')

        axes[2].imshow(image_array_outputs[0, :, :], cmap="gray")
        axes[2].set_title("output")
        axes[2].axis('off')

        plt.show()

    widgets.interact(plot_image, i=widgets.IntSlider(min=0, max=depth_slice.shape[2]-1, step=1, value=22))


# Train model
num_epochs = 300
# num_epochs =50
loss_evolution = []

# Guarda el tiempo de inicio
inicio = time.time()


loss_evolution_train = []
count = 0
for epoch in range(num_epochs):
    
    model.train()
    for batch in train_dataloader:
        
        inputs, targets = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        out =  nn.functional.sigmoid(outputs)
        # out = nn.functional.softmax(outputs, dim = 0)
        # one_hot_targets = F.one_hot(targets, num_classes=6)

        
        loss = dicefocall_loss_function(outputs, targets)
        loss.backward()
        
        optimizer.step()
        
        # plot_results(inputs,targets,outputs)
    loss_evolution_train.append(loss)
    # if count == 50:
    # plot_results(inputs,targets,out)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item()}")
        # count = 0
    # count = count + 1
    

print("Training complete.")
# Guarda el tiempo de finalización
fin = time.time()

# Calcula la diferencia
tiempo_total = fin - inicio

print(f"Tiempo de ejecución: {tiempo_total} segundos")



torch.save(model.state_dict(), "/home/paulagmtz/TFM_PAULA_24/MONAI-UNET/UNET_TRUE_MULTILABEL_1_7_model_200_dicefocall_loss.pth")





# def euclidean_distance(predictions, targets):
#     print(predictions)
#     print(targets)
#     print(predictions.shape)
#     print(targets.shape)
    
#     # Aplanar los tensores
#     predictions_flat = predictions.view(2, -1)
#     targets_flat = targets.view(2, -1)
#     dist = torch.norm(targets_flat-predictions_flat,dim=1)
#     return dist
# def validate_model(model, val_dataloader, device):
#     model.eval()  # Modo de evaluación

#     # Definir la función de pérdida
#     loss_function = DiceLoss(sigmoid=True, squared_pred=True)

#     # Variables para almacenar la evolución de la métrica
#     metric_history = []
#     loss_evolution = []

#     with torch.no_grad():
#         for batch in val_dataloader:
#             inputs, targets = batch[0].to(device), batch[1].to(device)
        
#             # Realizar predicciones
#             outputs = model(inputs)
            
#             one_hot_targets = F.one_hot(targets, num_clases=6)

#             # Calcular la pérdida
#             loss = loss_function(outputs, targets)

#             # caluclar la distancia euclidea
#             dist = euclidean_distance(outputs, targets)
#             # Guardar la métrica
#             loss_evolution.append(loss.item())
#             metric_history.append(dist)

#             # Mostrar imágenes de inputs, targets y outputs
#             plot_results(inputs, targets, outputs)

#     # Calcular la métrica promedio
#     average_metric = sum(metric_history) / len(metric_history)

#     print(f'Average Metric: {average_metric}')

#     # Guardar la evolución de la métrica en un archivo (por ejemplo, un archivo de texto)
#     with open("metric_history.txt", "w") as f:
#         for metric in metric_history:
#             f.write(f"{metric}\n")

    
    
# # Realizar la validación
#     validate_model(model, val_dataloader, device)
