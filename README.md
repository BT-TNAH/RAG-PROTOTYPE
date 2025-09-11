# RAG-PROTOTYPE
Prototype de RAG réalisé dans le cadre d'un stage de 6 mois en master 2 TNAH

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
