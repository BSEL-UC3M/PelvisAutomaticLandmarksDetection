#%%
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from UNETR import UNETR  
from dataset import PelvisDataset  
import monai
from monai.losses import DiceLoss 
from monai.metrics import MSEMetric 
from torchmetrics import Accuracy
from monai.metrics import DiceMetric

# Función para cargar el modelo y realizar la validación


def euclidean_distance(predictions, targets):
    # print(predictions)
    # print(targets)
    # print(predictions.shape)
    # print(targets.shape)
    
    # Aplanar los tensores
    predictions_flat = predictions.view(2, -1)
    targets_flat = targets.view(2, -1)
    dist = torch.norm(targets_flat-predictions_flat,dim=1)
    return dist
def validate_model(model, val_dataloader, device):
    model.eval()  # Modo de evaluación

    # Definir la función de pérdida
    loss_function = DiceLoss(sigmoid=True, squared_pred=True)

    # Variables para almacenar la evolución de la métrica
    metric_history = []
    loss_evolution = []
    
    accuracy_evolution = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
        
            # Realizar predicciones
            outputs = model(inputs)
            out =  nn.functional.sigmoid(outputs)

            # Calcular la pérdida
            loss = loss_function(outputs, targets)
            
            accuracy = Accuracy(task="binary").to(device)
            acc = accuracy(out, targets)
            # dice = DiceMetric(outputs, targets)

            # caluclar la distancia euclidea
            dist = euclidean_distance(out, targets)
            # Guardar la métrica
            loss_evolution.append(loss.item())
            metric_history.append(dist)
            accuracy_evolution.append(acc)

            # Mostrar imágenes de inputs, targets y outputs
            plot_results(inputs, targets, out)

    # Calcular la métrica promedio
    average_metric = sum(metric_history) / len(metric_history)
    average_accuracy = sum(accuracy_evolution)/(len(accuracy_evolution))
    print(f'Average Metric: {average_metric}')
    print(f'Average ACCURACY: {average_accuracy}')
    # loss_evolution_cpu = [tensor.cpu() for tensor in loss_evolution]  # Mover cada tensor a la CPU
    # loss_evolution_numpy = [tensor.detach().numpy() for tensor in loss_evolution]  # Convertir cada tensor a un array de NumPy
    # loss_evolution_numpy = [tensor.numpy() for tensor in loss_evolution]
    plot_loss_evolution(loss_evolution)


    # Guardar la evolución de la métrica en un archivo (por ejemplo, un archivo de texto)
    with open("metric_history.txt", "w") as f:
        for metric in metric_history:
            f.write(f"{metric}\n")
    return loss_evolution, average_metric, average_accuracy

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

if __name__ == "__main__":
    # Cargar el modelo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = UNETR(in_channels=1, out_channels=1, img_size=(112, 160, 208), feature_size=32, norm_name='batch')
    # model.to(device)
    # model.load_state_dict(torch.load("/home/paulagmtz/TFM_PAULA_24/MONAI-UNET/UNETR.pth"))

    # Cargar el conjunto de datos de validación
    root_dir = "/home/paulagmtz/TFM_PAULA_24/DATA_TFM/val_files"
    input_dim = (194, 148, 105)
    n_labels = 2
    val_dataset = PelvisDataset(root_dir, input_dim, n_labels, transform=None)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    # Realizar la validación
    loss_evolution, average_metric, average_accuracy = validate_model(model, val_dataloader, device)
    
    
    
