# Description

```
RAG-prototype/
├── README.md [ce fichier]
├── src/ [contient les scripts python]
│   ├── questions.csv
│   ├── main.py                     # Point d'entrée principal
│   ├── config/
│   │   └── configuration.py        # Module de configuration
│   ├── data_loading/
│   │   └── pdf_loader.py           # Module de chargement
│   ├── processing/
│   │   ├── chunking.py             # Module de chunking
│   │   └── embedding.py            # Module d'embedding
│   ├── storage/
│   │   └── vector_store.py         # Module de stockage
│   ├── retrieval/
│   │   └── retriever.py            # Module de retrieval
│   ├── generation/
│   │   └── generator.py            # Module de génération
│   ├── evaluation/
│   │   └── evaluator.py            # Module d'évaluation
│   ├── export/
│   │   └── csv_writer.py           # Module d'export
│   └── interface/
│       ├── evaluation_mode.py      # Mode évaluation
│       └── chatbot_mode.py         # Mode chatbot
├── data/ [pour des données de tests]
│   └── [19 fichiers PDF des transcriptions]
└── requirements.txt

```

## Installation et Configuration

### Prérequis

1. **Installer Ollama**
    
    Téléchargez et installez Ollama depuis [https://ollama.ai](https://ollama.ai/) selon votre système d'exploitation.
    
2. **Ouvrir l'invite de commande de votre ordinateur**

### Configuration de l'environnement

1. **Créer un environnement virtuel conda**
    
    Créer un environnement virtuel conda appelé par exemple "prototype" avec la bonne version de python (version 3.10) :
    
    ```bash
    conda create --name prototype python=3.10
    
    ```
    
2. **Activer cet environnement virtuel**
    
    ```bash
    conda activate prototype
    
    ```
    

### Installation des dépendances

1. **Installer les packages dans l'ordre strict suivant :**
    
    ```bash
    pip install llama-index-core==0.10.0
    
    ```
    
    ```bash
    pip install llama-index==0.10.0
    
    ```
    
    ```bash
    pip install llama-index-llms-ollama==0.1.0
    
    ```
    
    ```bash
    pip install llama-index-embeddings-ollama==0.1.0
    
    ```
    
    ```bash
    pip install faiss-cpu==1.7.4
    
    ```
    
    ```bash
    pip install pandas==2.1.0 numpy==1.24.0
    
    ```
    
    ```bash
    pip install PyPDF2==3.0.1 pymupdf==1.23.0 ollama==0.2.0 python-dotenv==1.0.0 tqdm colorama==0.4.6
    
    ```
    
2. **Télécharger PyTorch pour FAISS GPU et embeddings :**
    
    ```bash
    pip install torch
    
    ```
    

### Vérification de l'installation

1. **Vérifier que tous les packages sont bien installés :**
    
    ```bash
    python -c "import pandas as pd; import numpy as np; import llama_index; print('Installation réussie')"
    
    ```
    
2. **Tester que PyTorch est bien installé :**
    
    ```bash
    python -c "import torch; print(torch.__version__)"
    
    ```
    

### Configuration du projet

1. **Ouvrir le projet dans VS Code**
    
    Ouvrir une nouvelle fenêtre dans l'éditeur de code VS Code et ouvrir le dossier RAG-prototype.
    
2. **Activer l'environnement dans le terminal VS Code**
    
    Dans le Terminal de l'éditeur de code VS Code, écrire :
    
    ```bash
    conda activate prototype
    
    ```
    

### Installation des modèles Ollama

1. **Installer le modèle de langage**
    
    Installer le modèle de langage que vous souhaitez utiliser grâce à Ollama, par exemple :
    
    ```bash
    ollama pull llama3.1:latest
    
    ```
    
2. **Installer le modèle d'embedding**
    
    Installer le modèle d'embedding que vous souhaitez utiliser grâce à Ollama, par exemple :
    
    ```bash
    ollama pull nomic-embed-text:latest
    
    ```
    

## Utilisation

**Lancer le projet :**

Lancer `main.py` en cliquant en haut à droite sur la flèche dans VS Code, ou en utilisant :

```bash
python src/main.py

```

Pour sortir du programme, choisir l'option 3 dans le menu.
