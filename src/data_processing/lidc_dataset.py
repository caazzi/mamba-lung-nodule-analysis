import os
import glob
import pydicom
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    ToTensord,
)
from monai.data import Dataset, DataLoader

# Define a transformação para o pré-processamento de dados
# Esta é uma versão inicial e precisará ser expandida para incluir
# a extração de patches e aumento de dados.
def get_preprocessing_transforms(keys=["image"]):
    """
    Define as transformações de pré-processamento para os dados LIDC.
    Inclui normalização HU, reamostragem e adição de canal.
    """
    return Compose(
        [
            LoadImaged(keys=keys),  # Carrega a imagem DICOM
            AddChanneld(keys=keys),  # Adiciona um canal (para compatibilidade com MONAI)
            Orientationd(keys=keys, axcodes="RAS"),  # Padroniza a orientação do volume
            # Reamostragem para espaçamento isotrópico (ex: 1mm³)
            # O 'mode' pode ser ajustado para 'bilinear', 'bicubic', etc.
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            # Normalização de Unidades Hounsfield (HU) para uma faixa específica
            # Valores comuns para pulmão: -1000 a 400 HU
            ScaleIntensityRanged(
                keys=keys,
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # CropForegroundd(keys=keys, source_key="image", select_fn=lambda x: x > 0, margin=0),
            ToTensord(keys=keys), # Converte para tensor PyTorch
        ]
    )

class LIDCVolumeDataset(Dataset):
    """
    Dataset para carregar volumes de TC do LIDC.
    Esta classe é um esqueleto e precisará ser adaptada para lidar com
    as anotações de nódulos e a divisão de dados em nível de paciente.
    """
    def __init__(self, data_dir, transform=None):
        """
        Inicializa o dataset LIDC.

        Args:
            data_dir (str): Caminho para o diretório raiz dos dados LIDC (onde estão as pastas de pacientes).
            transform (callable, optional): Transformações a serem aplicadas aos dados.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = self._load_data_paths()

    def _load_data_paths(self):
        """
        Carrega os caminhos de todos os volumes DICOM.
        Assume que cada subpasta em data_dir é um estudo de paciente.
        """
        image_paths = []
        # Exemplo: data_dir/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1099.1.1.xxxx/sxxxx/
        # Você precisará adaptar isso para a estrutura exata do seu download LIDC.
        # Geralmente, os arquivos DICOM estão em subdiretórios de séries.

        # Este é um placeholder. Você precisará de uma lógica mais robusta
        # para encontrar todos os arquivos DICOM de um volume.
        # Uma abordagem comum é iterar sobre as pastas de pacientes e séries.

        # Exemplo simplificado para encontrar algumas imagens (ajuste conforme a estrutura real):
        # Para um dataset real, você precisaria de um parser mais robusto para DICOM
        # que lida com múltiplos arquivos por série e metadados.

        # Supondo que 'data_dir' contém pastas de pacientes, e dentro delas, séries com arquivos .dcm
        for patient_folder in os.listdir(self.data_dir):
            patient_path = os.path.join(self.data_dir, patient_folder)
            if os.path.isdir(patient_path):
                # Encontrar todas as séries dentro da pasta do paciente
                # Esta parte é complexa e depende da organização exata do LIDC.
                # Você pode precisar de um script separado para organizar os DICOMs em volumes.
                # Por simplicidade, vamos procurar por qualquer arquivo .dcm dentro da pasta do paciente
                # (o que não é ideal para volumes, mas serve como ponto de partida).
                dicom_files = glob.glob(os.path.join(patient_path, '**', '*.dcm'), recursive=True)
                if dicom_files:
                    # Para cada conjunto de arquivos DICOM que forma um volume,
                    # você criaria uma entrada aqui.
                    # Por enquanto, vamos apenas adicionar o caminho da pasta do paciente
                    # e assumir que a transformação lidará com a leitura dos DICOMs.
                    image_paths.append({"image": patient_path}) # MONAI LoadImaged pode lidar com pastas de DICOM

        if not image_paths:
            print(f"Aviso: Nenhuma imagem DICOM encontrada em {self.data_dir}. Verifique o caminho e a estrutura dos dados.")
            print("Esperado: data_dir/LIDC-IDRI-XXXX/SERIES_UID/DICOM_FILES.dcm")

        return image_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict

if __name__ == "__main__":
    # Exemplo de uso:
    # Crie uma pasta 'data/raw' e coloque alguns arquivos DICOM de teste ou o LIDC lá.
    # Exemplo de estrutura: data/raw/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1099.1.1.xxxx/sxxxx/*.dcm

    # Certifique-se de que 'data_dir' aponta para o diretório onde você baixou o LIDC
    # Por exemplo, se você baixou para 'data/raw/LIDC-IDRI-0001'

    # Crie as pastas se não existirem
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Substitua pelo caminho real onde você baixou o LIDC
    # Ex: lidc_data_root = "data/raw"
    # Para testar, você pode precisar de um pequeno conjunto de dados DICOM ou mockar alguns arquivos.
    # Por exemplo, se você tiver uma pasta 'data/raw/sample_patient_001' com arquivos .dcm
    lidc_data_root = "data/raw" # Ajuste este caminho

    # Obtenha as transformações
    transforms = get_preprocessing_transforms()

    # Crie o dataset
    # Note: LIDCVolumeDataset precisa de uma lógica mais robusta para carregar volumes
    # a partir de pastas de arquivos DICOM. MONAI's LoadImaged pode ajudar, mas
    # a estrutura de 'image_paths' precisa ser correta.
    # Para o LIDC, geralmente você terá um arquivo XML de anotações por paciente
    # e várias séries de imagens.
    try:
        dataset = LIDCVolumeDataset(data_dir=lidc_data_root, transform=transforms)
        if len(dataset) > 0:
            print(f"Dataset criado com {len(dataset)} volumes potenciais.")
            # Exemplo de carregamento do primeiro item
            sample_data = dataset[0]
            print(f"Shape do volume pré-processado: {sample_data['image'].shape}")
            print(f"Tipo de dado: {sample_data['image'].dtype}")
        else:
            print("Nenhum volume encontrado no dataset. Verifique o caminho e a estrutura dos dados.")
    except Exception as e:
        print(f"Ocorreu um erro ao criar ou acessar o dataset: {e}")
        print("Certifique-se de que a pasta 'data/raw' contém dados DICOM válidos do LIDC.")

    # Exemplo de DataLoader
    # dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    # for batch in dataloader:
    #     print(f"Batch shape: {batch['image'].shape}")
    #     break
