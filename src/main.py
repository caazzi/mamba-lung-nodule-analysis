import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from monai.transforms import Compose, LoadImaged, AddChanneld, Orientationd, Spacingd, ScaleIntensityRanged, ToTensord
from monai.metrics import ROCAUCMetric
from monai.losses import DiceLoss, BceDiceLoss
from tqdm import tqdm
import os

# Importar seus modelos e dataset
from src.data_processing.lidc_dataset import LIDCVolumeDataset, get_preprocessing_transforms
from src.models.cnn_baseline import CNNBaseline
from src.models.mamba_model import MambaModel3D # Lembre-se que MambaModel3D é um esqueleto

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name="model"):
    """
    Função para treinar o modelo.
    """
    model.to(device)
    best_val_auc = -1.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_labels = []
        train_outputs = []

        print(f"Epoch {epoch+1}/{num_epochs} - Training...")
        for batch_data in tqdm(train_loader):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Para classificação binária, as labels devem ser float32
            loss = criterion(outputs.squeeze(1), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            train_labels.extend(labels.cpu().numpy())
            train_outputs.extend(torch.sigmoid(outputs).cpu().numpy()) # Aplicar sigmoid para AUC

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")

        # Validação
        model.eval()
        val_labels = []
        val_outputs = []
        val_loss = 0.0

        with torch.no_grad():
            print(f"Epoch {epoch+1}/{num_epochs} - Validation...")
            for batch_data in tqdm(val_loader):
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                outputs = model(inputs)

                loss = criterion(outputs.squeeze(1), labels.float())
                val_loss += loss.item() * inputs.size(0)

                val_labels.extend(labels.cpu().numpy())
                val_outputs.extend(torch.sigmoid(outputs).cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader.dataset)

        # Calcular AUC
        # Certifique-se de que ROCAUCMetric recebe tensores e não numpy arrays
        # E que as saídas são probabilidades (após sigmoid)
        auc_metric = ROCAUCMetric()
        auc_metric(y_pred=torch.tensor(val_outputs), y=torch.tensor(val_labels))
        val_auc = auc_metric.aggregate().item()
        auc_metric.reset() # Resetar para a próxima época

        print(f"Epoch {epoch+1} Validation Loss: {val_epoch_loss:.4f}, Validation AUC: {val_auc:.4f}")

        # Salvar o melhor modelo
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            os.makedirs("experiments/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"experiments/checkpoints/{model_name}_best_model.pth")
            print(f"Melhor modelo salvo com AUC: {best_val_auc:.4f}")

def main():
    # Configurações
    data_dir = "data/raw" # Caminho para o diretório raiz dos dados LIDC
    batch_size = 1 # Ajuste conforme sua GPU e memória
    num_epochs = 5 # Número de épocas para treinamento (para teste inicial)
    learning_rate = 1e-4

    # Use GPU se disponível, caso contrário CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Pré-processamento dos Dados
    # As transformações devem ser adaptadas para incluir anotações de nódulos
    # e extração de patches. Para este exemplo, apenas o volume completo.
    # LIDCVolumeDataset precisa de um mecanismo para carregar anotações de nódulos
    # e associá-las aos volumes.

    # Para este exemplo, vamos criar dados dummy para simular o LIDCVolumeDataset
    # pois o carregamento real do LIDC é complexo e depende da estrutura dos arquivos.
    # Em um projeto real, você precisaria de um parser robusto para os arquivos XML do LIDC.

    # Simulação de dados para teste:
    # Cada item no dataset deve ser um dicionário com "image" e "label"
    # O LIDC-IDRI tem anotações de nódulos, então o "label" seria a malignidade
    # ou a presença/ausência de um nódulo.

    # Dummy data generation (replace with actual LIDC data loading)
    # This assumes you have a way to get image paths and corresponding labels.
    # For LIDC, labels would come from the XML annotations.

    # Example of how data_dicts should look like for MONAI:
    # data_dicts = [{"image": "path/to/dicom_folder_1", "label": 0},
    #               {"image": "path/to/dicom_folder_2", "label": 1}, ...]

    # For a quick test, let's create some dummy data dictionaries
    # In a real scenario, you'd parse LIDC XMLs to get labels.
    # For now, let's assume labels are randomly assigned for demonstration.

    # NOTE: The LIDCVolumeDataset needs to be robust enough to handle the actual LIDC structure.
    # The current LIDCVolumeDataset in lidc_dataset.py is a placeholder.

    # To run this main.py, you'll need to either:
    # 1. Implement the full LIDCVolumeDataset with label parsing.
    # 2. Create a small dummy dataset in 'data/raw' with a few DICOM series and corresponding dummy labels.

    # For demonstration, let's create a dummy dataset with dummy labels.
    # In a real scenario, `LIDCVolumeDataset` would load actual LIDC paths and parse XMLs for labels.

    # Dummy data for demonstration purposes (replace with actual LIDC loading)
    # This is highly simplified and assumes you have some DICOMs in data_dir.
    # You would need to map DICOM series to their corresponding malignancy labels from LIDC XMLs.
    # For now, we'll just use a dummy label (0 or 1) for each "volume".

    # The LIDCVolumeDataset as provided will try to find folders with DICOMs.
    # Let's assume you have some patient folders in `data_dir`

    # Get transforms for image and label (if labels are also volumes, like segmentation masks)
    # For classification, only image transform is needed, label is a scalar.
    transforms = get_preprocessing_transforms(keys=["image"])

    # Create the full dataset instance
    full_dataset = LIDCVolumeDataset(data_dir=data_dir, transform=transforms)

    if len(full_dataset) == 0:
        print("Erro: Nenhum dado encontrado no dataset. Certifique-se de que 'data/raw' contém dados LIDC.")
        print("Não é possível continuar o treinamento sem dados.")
        return

    # Divisão do dataset (treino/validação)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. Arquitetura do Modelo
    # Escolha qual modelo treinar: CNNBaseline ou MambaModel3D
    # model = CNNBaseline(in_channels=1, num_classes=1)
    model = MambaModel3D(in_channels=1, num_classes=1) # Usando o esqueleto Mamba

    # Função de Perda e Otimizador
    # Para classificação binária, BCEWithLogitsLoss é robusta.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 3. Treinamento e Validação
    print(f"Iniciando treinamento do modelo {model.__class__.__name__}...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name=model.__class__.__name__)

    print("\nTreinamento concluído!")
    print("O melhor modelo foi salvo em experiments/checkpoints/")

if __name__ == "__main__":
    main()
