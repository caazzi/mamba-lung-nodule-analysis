# Arquitetura Mamba para Análise de Nódulos Pulmonares em Tomografias Computadorizadas do LIDC

## Visão Geral do Projeto

Este projeto de pesquisa visa investigar a eficácia e eficiência da arquitetura Mamba para a análise de nódulos pulmonares em tomografias computadorizadas (TCs), com foco no vasto dataset público LIDC (Lung Image Database Consortium). O objetivo é comparar o desempenho do Mamba com uma arquitetura CNN 3D de referência, contribuindo para o avanço do diagnóstico precoce do câncer de pulmão.

## Justificativa

O câncer de pulmão é uma das principais causas de morte por câncer globalmente, e a detecção precoce de nódulos pulmonares em TCs é crucial para um melhor prognóstico. Arquiteturas de aprendizado profundo como CNNs e Vision Transformers (ViTs) têm avançado na área, mas apresentam limitações (foco local para CNNs, alta complexidade computacional para ViTs). A arquitetura Mamba, baseada em Modelos de Espaço de Estados Seletivos (SSMs), surge como uma alternativa promissora, oferecendo escalabilidade linear e modelagem eficaz de dependências de longo alcance.

## Objetivos

### Objetivo Geral:
Investigar a eficácia e eficiência da arquitetura Mamba para análise de nódulos pulmonares, com foco no dataset LIDC, comparando-a com uma baseline CNN.

### Objetivos Específicos:
1.  Adaptar e implementar um modelo Mamba (ex: Vision Mamba - Vim) para processamento de volumes 3D de TC do LIDC.
2.  Treinar e validar o modelo Mamba no LIDC, otimizando hiperparâmetros e utilizando métricas de avaliação adequadas (AUC, F1-Score, Sensibilidade, Especificidade).
3.  Comparar o desempenho (acurácia, custo computacional) do Mamba otimizado com uma arquitetura CNN 3D de referência (ex: 3D ResNet) no LIDC.

## Estrutura do Repositório

├── README.md├── requirements.txt├── .gitignore├── data/│   ├── raw/              # Dados brutos do dataset LIDC│   └── processed/        # Dados pré-processados (normalizados, reamostrados, patches)├── notebooks/            # Jupyter notebooks para exploração de dados e testes├── src/│   ├── models/           # Definições das arquiteturas dos modelos (Mamba, CNN)│   ├── data_processing/  # Scripts para pré-processamento e carregamento de dados│   ├── utils/            # Funções auxiliares (métricas, visualizações)│   └── main.py           # Script principal para treinamento e avaliação└── experiments/          # Logs de treinamento, modelos salvos e resultados
## Configuração do Ambiente

### Pré-requisitos
* Python 3.8+
* Git

### Passos para Configuração
1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/caazzi/mamba-lung-nodule-analysis.git](https://github.com/caazzi/mamba-lung-nodule-analysis.git)
    cd mamba-lung-nodule-analysis
    ```
2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    # No Windows:
    .\venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset LIDC-IDRI

O dataset LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative) pode ser baixado do [Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/datasets/lidc-idri/). Devido ao seu tamanho, o download pode ser demorado. Após o download, coloque os dados brutos na pasta `data/raw/`.

## Progresso e Próximos Passos

* **Mês 1:** Revisão de literatura aprofundada. Configuração do ambiente. Download e pré-processamento inicial do dataset LIDC.
* **Mês 2:** Implementação e início do treinamento do modelo CNN baseline. Início da adaptação/implementação do modelo Mamba.

## Referências

* **Mamba:** Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv preprint arXiv:2312.00752.
* **Vision Mamba (Vim):** Zhu, L., Zhang, Z., Wang, Y., & She, D. (2024). Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model. arXiv preprint arXiv:2401.09417.
* **ResNet:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
* **LIDC-IDRI:** Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., & Kazerooni, E. A. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans. Medical physics, 38(2), 915-931.

---

**Autor:** Claudio Amaral Azzi
**Orientador:** Prof. Dr. Wallace Correa de Oliveira Casaca
