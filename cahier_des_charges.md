# Cahier des charges

# Cahier des charges - Prototype RAG pour Archives Orales d’une grande entreprise

## 1. Vue d'ensemble du projet

### 1.1 Objectif principal

Développer un prototype de RAG (Retrieval-Augmented Generation) en Python pour interroger efficacement un corpus de 19 transcriptions d'archives orales concernant l'histoire de l'entreprise client de PH. Le prototype doit permettre l'évaluation automatisée de différents paramétrages et proposer un mode interactif pour des questions libres.

### 1.2 Contexte d'utilisation

- **Utilisateur cible** : Consultante/chercheuse en histoire appliquée
- **Objectif métier** : Accélérer l'exploitation des archives orales pour la production de mémoires historiques et films d'entreprise
- **Contrainte temporelle** : Optimiser le temps de recherche dans le corpus documentaire

## 2. Architecture générale du système

### 2.1 Structure des dossiers

```
RAG-prototype/
├── src/
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
├── data/
│   └── [19 fichiers PDF des transcriptions]
└── requirements.txt
```

### 2.2 Technologies imposées

- **Langage** : Python
- **Framework RAG** : LlamaIndex
- **LLM local** : Ollama
- **Vector Store** : Faiss Vector Store
- **Environnement** : Microsoft VS Code
- **Interface** : Terminal/Invite de commande Windows
- **Type de RAG** : Naïve RAG (architecture simple et modulaire)

## 3. Spécifications fonctionnelles

### 3.1 Mode 1 : Évaluation sur les 28 questions de test

### 3.1.1 Fonctionnement

- Lecture automatique du fichier `questions.csv` (28 questions prédéfinies)
- Traitement séquentiel de chaque question
- Génération d'un output complété dans le même fichier CSV

### 3.1.2 Structure du fichier d'entrée (colonnes A-F pré-remplies)

- **Colonne A** : Numero_identifiant (1-28)
- **Colonne B** : Question
- **Colonne C** : Interet_scientifique_question
- **Colonne D** : Typologie_question
- **Colonne E** : Source_idéale_chercheuse
- **Colonne F** : Réponse_idéale_chercheuse

### 3.1.3 Structure de l'output (nouvelles colonnes G-J pour chaque exécution)

Pour chaque lancement, ajouter 4 nouvelles colonnes :

- **Colonne G** : En-tête format "modèle_LLM_chunksize_chunkoverlap_modèle_embedding_topk"
    - Contenu : Sources trouvées par retrieval format "[fichier_page, fichier_page, ...]" pour les sources provenant des contenus textuels et format "[fichier_meta, fichier_meta, ...]" pour les sources provenant des métadonnées des fichiers. Les deux peuvent alterner, être mélangés : "[fichier_meta, fichier_page, fichier_page, fichier_page, fichier_meta ...]
- **Colonne H** : En-tête identique à G
    - Contenu : Score F1 (évaluation automatique de conformité entre source idéale et source trouvée)
- **Colonne I** : En-tête identique à G
    - Contenu : Réponse générée par le chatbot avec sources citées basées sur celles trouvées par retrieval dans la colonne G.
- **Colonne J** : En-tête identique à G
    - Contenu : Vide (à remplir manuellement par la chercheuse)
- **Colonne K** :  En-tête identique à G
    - Contenu : Sources citées dans la Réponse générée par le chatbot issues des sources trouvées par retrieval de la colonne G (format "[fichier_page, fichier_page, ...]" pour les sources provenant des contenus textuels et format "[fichier_meta, fichier_meta, ...]" pour les sources provenant des métadonnées des fichiers. Les deux peuvent alterner, être mélangés : "[fichier_meta, fichier_page, fichier_page, fichier_page, fichier_meta ...])
- **Colonne L** :  En-tête identique à G
    - Contenu : Score F1 (évaluation automatique de conformité entre source idéale et source citées dans la génération de la réponse)

### 3.2 Mode 2 : Chatbot interactif pour questions libres

### 3.2.1 Fonctionnement

- Interface en ligne de commande
- Questions libres une par une (pas de batch)
- Réponses avec citation des sources (fichier, page/meta, passage)
- Session interactive continue jusqu'à arrêt utilisateur

### 3.2.2 Format des réponses

- Réponse générée par le LLM
- Sources citées : nom du fichier PDF, numéro de page, extrait pertinent
- Pas d'évaluation automatique (vérification manuelle par l'utilisateur)

## 4. Spécifications techniques

### 4.1 Paramétrage configurable à chaque lancement

### 4.1.1 Interface de paramétrage

- Sélection interactive via Terminal/Invite de commande
- Pas d'interface graphique (Frontend)
- Validation des paramètres avant lancement

### 4.1.2 Modèles de LLM (2 options)

- `llama3.1:latest` (priorité, déjà testé)
- `llama3.3` (option secondaire)

### 4.1.3 Chunk Size (10 options)

- **Petits chunks** (phrase/petit paragraphe) : 250, 300, 400
- **Chunks moyens** (paragraphe moyen/grand) : 500, 600
- **Grands chunks** (page) : 700, 800, 900, 1000, 2000

### 4.1.4 Chunk Overlap (12 options)

- **Nul** : 0
- **Petit** (pour phrases) : 10, 20, 30
- **Moyen** (pour paragraphes) : 50, 60, 70, 80, 90, 100
- **Grand** (pour pages) : 800, 2000

### 4.1.5 Modèles d'embedding (4 options prioritaires)

- `nomic-embed-text:latest` (priorité, déjà testé)
- `bge-large:latest`
- `granite-embedding`
- `all-minilm:l6-v2`

### 4.1.6 Similarity Top-K (7 options)

- **Petit** : 3, 4, 5
- **Grand** : 20, 30, 40, 50

### 4.2 Architecture modulaire requise

### 4.2.1 Modules principaux

1. **Module de configuration** : Gestion des paramètres utilisateur
2. **Module de chargement** : Lecture et preprocessing des PDF
3. **Module de chunking** : Découpage des documents selon les paramètres
4. **Module d'embedding** : Vectorisation des chunks
5. **Module de stockage** : Gestion du vector store Faiss
6. **Module de retrieval** : Recherche de similarité
7. **Module de génération** : Interface avec le LLM
8. **Module d'évaluation** : Calcul du score F1
9. **Module d'export** : Écriture des résultats CSV

### 4.2.2 Flux de traitement

```
Configuration → Chargement PDF → Chunking → Embedding →
Stockage → [Retrieval → Génération → Évaluation] → Export

```

### 4.3 Traitement des documents PDF

### 4.3.1 Métadonnées personnalisées requises

Chaque PDF contient des métadonnées enrichies :

- **Customintervieweename** : Nom de l'interviewé
- **Custominterviewdate** : Date de l'entretien
- **Custominterviewlanguage** : Langue de l'entretien (français/anglais)
- **Customarchivetype** : Type d'archive

### 4.3.2 Exploitation des métadonnées

- Extraction automatique des métadonnées lors du chargement
- Utilisation pour le retrieval et la génération de réponses
- Intégration dans les citations des sources

### 4.4 Système d'évaluation automatique

### 4.4.1 Calcul du score F1

- D’un côté, comparaison entre sources idéales (colonne E) et sources trouvées d’un côté (colonne G), et de l’autre côté, comparaison entre sources idéales (colonne E) et sources citées dans la réponse générée (colonne L)
- Algorithme de matching basé sur les noms de fichiers et pages/meta
- Score numérique entre 0 et 1
- F1 Score spécial pour les questions “hors-sujet” attendant une réponse nulle.

### 4.4.2 Format des sources

- **Source idéale** : "fichier1_page1, fichier2_page2, fichier3_meta, fichier4_meta ..."
- **Source trouvée** : "[fichier1_page1, fichier2_page2, fichier3_meta, ...]"
- Normalisation nécessaire pour comparaison

## 5. Spécifications des données

### 5.1 Corpus documentaire

- **Nombre de fichiers** : 19 PDF
- **Taille moyenne** : ~20 pages par fichier
- **Langues** : 16 fichiers français, 3 fichiers anglais
- **Contenu** : Transcriptions d'entretiens relus et corrigés
- **Format** : PDF avec métadonnées enrichies

### 5.2 Types de questions supportées

### 5.2.1 Questions élémentaires

- **Intervieweur** : Reproduction de questions/réponses exactes des transcriptions
- **Hors-sujet** : Détection d'informations absentes (réponse négative attendue)

### 5.2.2 Questions intermédiaires

- **Métadonnées uniquement** : Recherche dans les métadonnées des fichiers
- **Texte + métadonnées** : Recherche combinée texte et métadonnées

### 5.2.3 Questions complexes

- **Informations simples** : Mots/expressions présents tels quels
- **Informations complexes** : Dates, années, informations calculées
- **Questions déductives** : Comparaisons et classements

## 6. Contraintes et exigences

### 6.1 Contraintes techniques

- **Confidentialité** : Traitement local uniquement (pas de cloud)
- **Performance** : Temps de réponse raisonnable pour 28 questions
- **Stabilité** : Gestion des erreurs et récupération
- **Reproductibilité** : Résultats cohérents pour mêmes paramètres

### 6.2 Contraintes d'usage

- **Simplicité** : Interface en ligne de commande uniquement
- **Flexibilité** : Tests rapides de différents paramétrages
- **Traçabilité** : Conservation de l'historique des tests dans le CSV

### 6.3 Livrables attendus

### 6.3.1 Code source

- Code Python modulaire et documenté
- Scripts de lancement pour les 2 modes
- Fichiers de configuration/paramétrage

### 6.3.2 Documentation

- Instructions d'installation et dépendances
- Guide d'utilisation des 2 modes
- Documentation des paramètres et leur impact

### 6.3.3 Résultats de tests

- Fichier questions.csv complété avec plusieurs paramétrages
- Analyse comparative des performances
- Recommandations de paramétrage optimal

## 7. Validation et tests

### 7.1 Tests fonctionnels

- Traitement correct des 28 questions de test
- Fonctionnement du mode interactif
- Exactitude des métadonnées extraites
- Cohérence des citations de sources

### 7.2 Tests de performance

- Temps de traitement pour différents paramétrages
- Qualité des réponses (évaluation manuelle sur échantillon)
- Stabilité sur sessions longues

### 7.3 Tests d'intégration

- Compatibilité avec tous les modèles requis
- Gestion des fichiers PDF avec métadonnées personnalisées
- Export correct vers CSV

## 8. Installation et déploiement

### 8.1 Prérequis système

- Python 3.x
- Ollama installé et configuré
- Modèles LLM et embedding téléchargés
- Dépendances Python (LlamaIndex, Faiss, pandas, etc.)

### 8.2 Installation

- Scripts d'installation automatisée des dépendances
- Vérification de la configuration Ollama
- Tests de connectivité aux modèles

### 8.3 Configuration initiale

- Placement des fichiers PDF dans le dossier data/
- Vérification des métadonnées des PDF
- Tests de base avec paramètres par défaut
