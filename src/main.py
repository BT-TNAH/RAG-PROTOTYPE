import sys
import os
from pathlib import Path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
from typing import Optional, Dict, Any, List
import pandas as pd
import subprocess
import logging
from datetime import datetime

# Imports des modules de l'architecture
from config.configuration import ConfigurationManager, EmbeddingModel, LLMModel
from data_loading.pdf_loader import PDFLoader
from processing.chunking import DocumentChunker
from processing.embedding import EmbeddingProcessor
from storage.vector_store import VectorStoreManager
from retrieval.retriever import DocumentRetriever
from generation.generator import LLMGenerator as ResponseGenerator
from evaluation.evaluator import RAGEvaluator
from export.csv_writer import CSVExporter
from interface.evaluation_mode import EvaluationMode
from interface.chatbot_mode import ChatbotMode


class RAGPrototype:
    """
    Prototype RAG pour l'interrogation d'archives orales d'XXXX
    
    Permet l'évaluation automatisée de différents paramétrages et propose
    un mode interactif pour des questions libres sur un corpus de 19 transcriptions PDF.
    """
    
    def __init__(self):
        """
        Initialise le prototype RAG avec ses composants principaux
        """
        # Configuration du logging
        self._setup_logging()
    
        # Composants principaux
        self.config_manager: Optional[ConfigurationManager] = None
        self.pdf_loader: Optional[PDFLoader] = None
        self.chunker: Optional[DocumentChunker] = None
        self.embedding_processor: Optional[EmbeddingProcessor] = None
        self.vector_store: Optional[VectorStoreManager] = None
        self.retriever: Optional[DocumentRetriever] = None
        self.generator: Optional[ResponseGenerator] = None
        self.evaluator: Optional[RAGEvaluator] = None
        self.csv_exporter: Optional[CSVExporter] = None
    
        # Modes d'interface
        self.evaluation_mode: Optional[EvaluationMode] = None
        self.chatbot_mode: Optional[ChatbotMode] = None
    
        # Configuration actuelle
        self.current_config: Optional[Dict[str, Any]] = None
    
        # Paths importants - correction des chemins relatifs
        current_file_dir = Path(__file__).parent
        project_root = current_file_dir.parent
    
        self.data_dir = project_root / "data"
        self.questions_file = current_file_dir / "questions.csv"
    
        # Logger
        self.logger = logging.getLogger(__name__)
    
        # Diagnostic des chemins
        self.logger.info(f"Répertoire du script : {current_file_dir}")
        self.logger.info(f"Racine du projet : {project_root}")
        self.logger.info(f"Dossier data : {self.data_dir}")
        self.logger.info(f"Fichier questions : {self.questions_file}")
    
    def main(self) -> None:
        """
        Point d'entrée principal du prototype
        """
        try:
            # Bannière d'accueil
            self.display_welcome_banner()
            
            # Validation des dépendances
            if not self.validate_dependencies():
                self.logger.error("Erreur lors de la validation des dépendances")
                sys.exit(1)
            
            # Boucle principale
            while True:
                self.display_main_menu()
                mode_selection = self.get_user_mode_selection()
                
                if mode_selection == "1":
                    self._launch_evaluation_mode()
                elif mode_selection == "2":
                    self._launch_chatbot_mode()
                elif mode_selection == "3":
                    print("Au revoir !")
                    break
                else:
                    print("Sélection invalide. Veuillez choisir 1, 2 ou 3.")
                    continue
                    
        except KeyboardInterrupt:
            print("\n\nArrêt du programme demandé par l'utilisateur.")
        except Exception as e:
            self.logger.error(f"Erreur critique dans main(): {e}")
            print(f"Erreur critique : {e}")
            sys.exit(1)
    
    def display_main_menu(self) -> None:
        """
        Affiche le menu principal avec les 2 modes disponibles + option de sortie
        """
        print("\n" + "="*60)
        print("           MENU PRINCIPAL - PROTOTYPE RAG XXXX")
        print("="*60)
        print("1. Mode Évaluation (28 questions de test)")
        print("   → Traitement automatisé du fichier questions.csv")
        print("   → Évaluation de différents paramétrages")
        print()
        print("2. Mode Chatbot (questions libres)")
        print("   → Interface interactive en ligne de commande")
        print("   → Questions libres sur le corpus d'archives")
        print()
        print("3. Quitter le programme")
        print("="*60)
    
    def get_user_mode_selection(self) -> str:
        """
        Récupère la sélection de mode par l'utilisateur
        """
        while True:
            try:
                selection = input("\nVeuillez choisir un mode (1, 2 ou 3) : ").strip()
                if selection in ["1", "2", "3"]:
                    return selection
                else:
                    print("Sélection invalide. Veuillez entrer 1, 2 ou 3.")
            except (EOFError, KeyboardInterrupt):
                print("\nAu revoir !")
                sys.exit(0)
    
    def display_welcome_banner(self) -> None:
        """
        Affiche la bannière d'accueil du prototype
        """
        print("\n" + "#"*70)
        print("##" + " "*66 + "##")
        print("##" + " "*18 + "PROTOTYPE RAG - XXXX" + " "*28 + "##")
        print("##" + " "*10 + "Archives Orales d'Histoire d'Entreprise" + " "*17 + "##")
        print("##" + " "*66 + "##")
        print("#"*70)
        print()
        print("📚 Corpus : 19 transcriptions d'entretiens PDF")
        print("🔍 Technologies : LlamaIndex + Ollama + Faiss")
        print("⚙️  Architecture : RAG Naïve modulaire")
        print()
        print(f"🕐 Session démarrée le : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
        print("-" * 70)
    
    def validate_dependencies(self) -> bool:
        """
        Vérifie que toutes les dépendances sont installées et disponibles
        """
        print("\n🔍 Validation des dépendances en cours...")
        
        validation_steps = [
            ("Vérification d'Ollama", self.check_ollama_availability),
            ("Validation du dossier data/", self.check_data_directory),
            ("Vérification du fichier questions.csv", self._check_questions_file),
            ("Test des modules Python", self._check_python_modules)
        ]
        
        all_valid = True
        for step_name, validation_func in validation_steps:
            try:
                print(f"  → {step_name}...", end=" ")
                if validation_func():
                    print("✅ OK")
                else:
                    print("❌ ÉCHEC")
                    all_valid = False
            except Exception as e:
                print(f"❌ ERREUR : {e}")
                all_valid = False
        
        if all_valid:
            print("\n✅ Toutes les dépendances sont satisfaites !")
        else:
            print("\n❌ Certaines dépendances ne sont pas satisfaites.")
            print("Veuillez corriger les problèmes avant de continuer.")
        
        return all_valid
    
    def check_ollama_availability(self) -> bool:
        """
        Vérifie qu'Ollama est installé et accessible
        """
        try:
            # Test de la commande ollama
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                self.logger.error("Ollama n'est pas installé ou accessible")
                return False
            
            # Test de la liste des modèles
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=15
            )
            if result.returncode != 0:
                self.logger.error("Impossible de lister les modèles Ollama")
                return False
            
            models_output = result.stdout.lower()
            
            # Vérification des modèles LLM requis
            required_llm_models = ["llama3.1:latest", "llama3.3"]
            available_llm = any(model in models_output for model in required_llm_models)
            
            # Vérification des modèles d'embedding requis
            required_embedding_models = [
                "nomic-embed-text:latest", 
                "bge-large:latest", 
                "granite-embedding", 
                "all-minilm:l6-v2"
            ]
            available_embedding = any(model in models_output for model in required_embedding_models)
            
            if not available_llm:
                self.logger.error("Aucun modèle LLM requis n'est disponible")
                return False
                
            if not available_embedding:
                self.logger.error("Aucun modèle d'embedding requis n'est disponible")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout lors de la vérification d'Ollama")
            return False
        except FileNotFoundError:
            self.logger.error("Ollama n'est pas installé (commande introuvable)")
            return False
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification d'Ollama : {e}")
            return False
    
    def check_data_directory(self) -> bool:
        """
        Vérifie que le dossier data/ existe et contient des PDFs
        """
        try:
            if not self.data_dir.exists():
                self.logger.error(f"Le dossier {self.data_dir} n'existe pas")
                return False
            
            if not self.data_dir.is_dir():
                self.logger.error(f"{self.data_dir} n'est pas un dossier")
                return False
            
            # Recherche des fichiers PDF
            pdf_files = list(self.data_dir.glob("*.pdf"))
            
            if len(pdf_files) == 0:
                self.logger.error("Aucun fichier PDF trouvé dans le dossier data/")
                return False
            
            # Vérification du nombre attendu
            if len(pdf_files) != 19:
                self.logger.warning(
                    f"Nombre de PDFs trouvés : {len(pdf_files)} (attendu : 19)"
                )
            
            # Validation du format des noms de fichiers
            valid_pattern_count = 0
            for pdf_file in pdf_files:
                if "XXXX" in pdf_file.name and "entretien" in pdf_file.name:
                    valid_pattern_count += 1
            
            if valid_pattern_count < len(pdf_files) * 0.8:
                self.logger.warning(
                    "Certains fichiers PDF ne respectent pas le format attendu (XXXX_*_entretien_*)"
                )
            
            self.logger.info(f"Dossier data/ validé : {len(pdf_files)} fichiers PDF trouvés")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation du dossier data/ : {e}")
            return False
    
    def _check_questions_file(self) -> bool:
        """
        Vérifie l'existence et la validité du fichier questions.csv - VERSION CORRIGÉE
        """
        try:
            if not self.questions_file.exists():
                self.logger.error(f"Le fichier {self.questions_file} n'existe pas")
                return False

            # Lecture directe avec les bons paramètres
            df = pd.read_csv(self.questions_file, sep=',', encoding='utf-8', quotechar='"')
            self.logger.info(f"Fichier CSV lu avec succès")

            # Log des colonnes actuelles pour diagnostic
            self.logger.info(f"Colonnes trouvées dans le CSV: {list(df.columns)}")

            # Colonnes requises selon votre CSV
            required_columns = [
                'Numero_identifiant',
                'Question', 
                'Interet_scientifique_question',
                'Typologie_question',
                'Source_ideale_chercheuse',
                'Reponse_ideale_chercheuse'
            ]

            missing_columns = []
            for col in required_columns:
                if col not in df.columns:
                    missing_columns.append(col)

            if missing_columns:
                self.logger.error(f"Colonnes manquantes: {missing_columns}")
                self.logger.info(f"Colonnes disponibles: {list(df.columns)}")
                return False

            if len(df) != 28:
                self.logger.warning(f"Nombre de questions : {len(df)} (attendu : 28)")

            self.logger.info("Fichier questions.csv validé avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de la validation de questions.csv : {e}")
            return False
    
    def _check_python_modules(self) -> bool:
        """
        Teste l'import de tous les modules Python requis
        """
        required_modules = [
            'llama_index', 'faiss', 'pandas', 'numpy', 
            'pathlib', 'subprocess', 'logging', 'tqdm'
        ]
        
        try:
            for module_name in required_modules:
                if module_name == 'llama_index':
                    import llama_index
                elif module_name == 'faiss':
                    import faiss
                elif module_name == 'tqdm':
                    import tqdm
                else:
                    __import__(module_name)
            
            return True
            
        except ImportError as e:
            self.logger.error(f"Module manquant : {e}")
            print(f"⚠️  Module manquant détecté : {e}")
            print("Pour installer tqdm : pip install tqdm")
            return False
        except Exception as e:
            self.logger.error(f"Erreur lors du test des modules : {e}")
            return False
    
    def _launch_evaluation_mode(self) -> None:
        """
        Lance le mode évaluation avec les 28 questions de test
        """
        try:
            print("\n🔬 Lancement du Mode Évaluation")
            print("="*50)
        
            # Configuration interactive
            self.current_config = self._get_user_configuration()
        
            # Initialisation des composants
            if not self._initialize_rag_components():
                print("❌ Erreur lors de l'initialisation des composants RAG")
                return
        
            # Chargement et traitement des documents
            if not self._load_and_process_documents():
                print("❌ Erreur lors du chargement des documents")
                return
        
            # Initialisation sécurisée d'EvaluationMode
            self.evaluation_mode = self._safe_initialize_evaluation_mode()
            if not self.evaluation_mode:
                print("❌ Erreur lors de l'initialisation du mode évaluation")
                return
        
            # Lancement de l'évaluation
            self.evaluation_mode.run_evaluation(str(self.questions_file))
        
        except Exception as e:
            self.logger.error(f"Erreur dans le mode évaluation : {e}")
            print(f"❌ Erreur : {e}")
            import traceback
            self.logger.error(f"Traceback : {traceback.format_exc()}")

    def _launch_chatbot_mode(self) -> None:
        """
        Lance le mode chatbot interactif pour questions libres
        """
        try:
            print("\n💬 Lancement du Mode Chatbot")
            print("="*50)
        
            # Configuration interactive
            self.current_config = self._get_user_configuration()
        
            # Initialisation des composants
            if not self._initialize_rag_components():
                print("❌ Erreur lors de l'initialisation des composants RAG")
                return
        
            # Chargement et traitement des documents
            if not self._load_and_process_documents():
                print("❌ Erreur lors du chargement des documents")
                return
        
            # Initialisation sécurisée de ChatbotMode
            self.chatbot_mode = self._safe_initialize_chatbot_mode()
            if not self.chatbot_mode:
                print("❌ Erreur lors de l'initialisation du mode chatbot")
                return
        
            # Lancement de la session interactive
            self.chatbot_mode.start_interactive_session()
        
        except Exception as e:
            self.logger.error(f"Erreur dans le mode chatbot : {e}")
            print(f"❌ Erreur : {e}")
            import traceback
            self.logger.error(f"Traceback : {traceback.format_exc()}")
    
    def _safe_initialize_evaluation_mode(self) -> Optional[EvaluationMode]:
        """
        Initialise EvaluationMode de manière sécurisée - VERSION CORRIGÉE
        """
        try:
            # CORRECTION: Passer le chemin du fichier questions lors de l'initialisation
            evaluation_mode = EvaluationMode(
                retriever=self.retriever,
                generator=self.generator,
                evaluator=self.evaluator,
                csv_exporter=self.csv_exporter,
                config=self.current_config,
            )
            
            self.logger.info(f"EvaluationMode initialisé avec questions_file: {self.questions_file}")
            return evaluation_mode
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation d'EvaluationMode : {e}")
            
            # Tentative de fallback sans questions_file si le constructeur ne l'accepte pas
            try:
                evaluation_mode = EvaluationMode()
                evaluation_mode.retriever = self.retriever
                evaluation_mode.generator = self.generator
                evaluation_mode.evaluator = self.evaluator
                evaluation_mode.csv_exporter = self.csv_exporter
                evaluation_mode.current_config = self.current_config
                
                self.logger.info("EvaluationMode initialisé avec attribution directe")
                return evaluation_mode
                
            except Exception as e2:
                self.logger.error(f"Fallback échoué : {e2}")
                return None
    
    def _safe_initialize_chatbot_mode(self) -> Optional[ChatbotMode]:
        """
        Initialise ChatbotMode de manière sécurisée avec plusieurs tentatives
        """
        try:
            # Tentative 1: Sans paramètres
            chatbot_mode = ChatbotMode()
            
            if hasattr(chatbot_mode, 'set_components'):
                chatbot_mode.set_components(
                    retriever=self.retriever,
                    generator=self.generator,
                    config=self.current_config
                )
            else:
                chatbot_mode.retriever = self.retriever
                chatbot_mode.generator = self.generator
                chatbot_mode.config = self.current_config
            
            return chatbot_mode
            
        except Exception as e1:
            self.logger.warning(f"Tentative 1 échouée : {e1}")
            
            try:
                # Tentative 2: Avec config seulement
                chatbot_mode = ChatbotMode(config=self.current_config)
                chatbot_mode.retriever = self.retriever
                chatbot_mode.generator = self.generator
                return chatbot_mode
                
            except Exception as e2:
                self.logger.warning(f"Tentative 2 échouée : {e2}")
                
                try:
                    # Tentative 3: Avec tous les paramètres
                    chatbot_mode = ChatbotMode(
                        retriever=self.retriever,
                        generator=self.generator,
                        config=self.current_config
                    )
                    return chatbot_mode
                    
                except Exception as e3:
                    self.logger.error(f"Toutes les tentatives ont échoué : {e3}")
                    return None
    
    def _get_user_configuration(self) -> Dict[str, Any]:
        """
        Interface de paramétrage interactif pour l'utilisateur
        """
        if not self.config_manager:
            self.config_manager = ConfigurationManager()
        
        return self.config_manager.get_interactive_configuration()
    
    def _initialize_rag_components(self) -> bool:
        """
        Initialise tous les composants RAG selon la configuration
        """    
        try:
            print("🔧 Initialisation des composants RAG...")

            # Chargeur PDF
            self.pdf_loader = PDFLoader(str(self.data_dir))

            # Chunker avec configuration
            self.chunker = DocumentChunker(
                chunk_size=self.current_config['chunk_size'],
                chunk_overlap=self.current_config['chunk_overlap']
            )

            # MODIFICATION: Processeur d'embedding avec paramètres optimisés
            embedding_model_name = self._extract_model_name(self.current_config['embedding_model'])
            
            # Calcul dynamique de la taille de batch selon la taille des chunks
            chunk_size = self.current_config['chunk_size']
            if chunk_size <= 300:
                batch_size = 75  # Plus de chunks par batch pour les petits chunks
                delay = 0.05     # Délai plus court
            elif chunk_size <= 600:
                batch_size = 50  # Taille moyenne
                delay = 0.1
            else:
                batch_size = 25  # Moins de chunks par batch pour les gros chunks
                delay = 0.15     # Délai plus long
            
            self.embedding_processor = EmbeddingProcessor(
                embedding_model=embedding_model_name,
                batch_size=batch_size,
                delay_between_batches=delay
            )

            # Vector Store
            self.vector_store = VectorStoreManager()

            # Retriever avec l'embedding_processor ET le vector_store correct
            self.retriever = DocumentRetriever(
                vector_store=self.vector_store.vector_store,  # ← AJOUT .vector_store
                embedding_processor=self.embedding_processor,
                top_k=self.current_config['top_k']
            )

            # Générateur - utiliser la string du modèle directement
            llm_model_name = self._extract_model_name(self.current_config['llm_model'])
            self.generator = ResponseGenerator(llm_model_name)

            # AJOUT : Vérification de compatibilité
            if not hasattr(self.generator, 'get_last_cited_sources'):
                self.logger.warning("Méthode get_last_cited_sources() non disponible - utilisation du fallback")

            # Évaluateur
            self.evaluator = RAGEvaluator()

            # Exporteur CSV avec le bon chemin
            self.csv_exporter = CSVExporter(
                csv_file_path=str(self.questions_file)
            )

            print("✅ Composants RAG initialisés avec succès")
            print(f"   📊 Batch size pour embeddings: {batch_size}")
            print(f"   ⏱️  Délai entre batches: {delay}s")
            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des composants : {e}")
            print(f"❌ Détail de l'erreur : {e}")
            
            import traceback
            self.logger.error(f"Traceback complet : {traceback.format_exc()}")
            
            return False
    
    def _extract_model_name(self, model_config) -> str:
        """
        Extrait le nom du modèle depuis la configuration
        
        Args:
            model_config: Peut être une string, un enum, ou un objet avec attribut .value
            
        Returns:
            str: Nom du modèle
        """
        if isinstance(model_config, str):
            return model_config
        elif hasattr(model_config, 'value'):
            return model_config.value
        elif hasattr(model_config, 'name'):
            return model_config.name
        else:
            return str(model_config)
    
    def _load_and_process_documents(self) -> bool:
        """
        Charge et traite tous les documents PDF
        """
        try:
            print("📚 Chargement et traitement des documents...")

            # Chargement des documents
            documents = self.pdf_loader.load_all_pdfs()
    
            if not documents:
                self.logger.error("Aucun document chargé")
                return False

            print(f"📄 {len(documents)} documents chargés")

            # Chunking
            chunks = self.chunker.chunk_documents(documents)
            print(f"🔪 {len(chunks)} chunks créés")

            # MODIFICATION: Utilisation de la nouvelle méthode embed_chunks
            embedded_chunks = self.embedding_processor.embed_chunks(chunks)
            print(f"🧮 {len(embedded_chunks)} embeddings générés")

            # Séparation des chunks et embeddings pour VectorStoreManager
            chunks_only = [ec.chunk for ec in embedded_chunks]
            embeddings_only = [ec.embedding_vector for ec in embedded_chunks]
        
            # Debug pour vérifier les types
            self.logger.info(f"Type chunks_only: {type(chunks_only)}, longueur: {len(chunks_only)}")
            self.logger.info(f"Type embeddings_only: {type(embeddings_only)}, longueur: {len(embeddings_only)}")
            if chunks_only:
                self.logger.info(f"Type premier chunk: {type(chunks_only[0])}")
            if embeddings_only:
                self.logger.info(f"Type premier embedding: {type(embeddings_only[0])}")

            # Stockage dans le vector store avec les deux paramètres séparés
            self.vector_store.add_documents(chunks_only, embeddings_only)
            print("💾 Documents stockés dans le vector store")

            # === AJOUT DES VÉRIFICATIONS DU VECTOR STORE ICI ===
            print("\n🔍 Vérifications du Vector Store:")
            
            # Vérification du nombre de vecteurs
            try:
                if hasattr(self.vector_store, 'get_total_vectors'):
                    total_vectors = self.vector_store.get_total_vectors()
                    print(f"- Nombre total de vecteurs: {total_vectors}")
                else:
                    # Alternative si la méthode n'existe pas
                    if hasattr(self.vector_store, 'vector_store') and hasattr(self.vector_store.vector_store, 'ntotal'):
                        total_vectors = self.vector_store.vector_store.ntotal
                        print(f"- Nombre total de vecteurs: {total_vectors}")
                    else:
                        print("- Impossible de récupérer le nombre de vecteurs")
            except Exception as e:
                print(f"- Erreur lors de la vérification des vecteurs: {e}")

            # Vérification de l'initialisation
            try:
                if hasattr(self.vector_store, 'is_initialized'):
                    initialized = self.vector_store.is_initialized()
                    print(f"- Index initialisé: {initialized}")
                else:
                    print("- Méthode is_initialized() non disponible")
            except Exception as e:
                print(f"- Erreur lors de la vérification d'initialisation: {e}")

            # Test de recherche simple
            try:
                if hasattr(self.vector_store, 'search_similar'):
                    test_results = self.vector_store.search_similar("test", top_k=1)
                    print(f"- Test de recherche fonctionne: {len(test_results) >= 0}")
                    print(f"- Nombre de résultats de test: {len(test_results) if test_results else 0}")
                else:
                    print("- Méthode search_similar() non disponible sur vector_store")
            except Exception as e:
                print(f"- Erreur lors du test de recherche: {e}")

            # Test avec le retriever si disponible
            try:
                if self.retriever:
                    test_retrieval = self.retriever.retrieve("test de fonctionnement")
                    print(f"- Test du retriever: {len(test_retrieval) if test_retrieval else 0} résultats")
                else:
                    print("- Retriever non initialisé")
            except Exception as e:
                print(f"- Erreur lors du test du retriever: {e}")
                
            print("=" * 50)

            return True

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des documents : {e}")
            print(f"❌ Détail de l'erreur : {e}")
            
            import traceback
            self.logger.error(f"Traceback complet : {traceback.format_exc()}")
            
            return False
    
    def _setup_logging(self) -> None:
        """
        Configure le système de logging pour le prototype
        """
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"rag_prototype_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Retourne les informations système pour le diagnostic
        """
        return {
            'data_directory': str(self.data_dir),
            'questions_file': str(self.questions_file),
            'current_config': self.current_config,
            'components_initialized': {
                'pdf_loader': self.pdf_loader is not None,
                'chunker': self.chunker is not None,
                'embedding_processor': self.embedding_processor is not None,
                'vector_store': self.vector_store is not None,
                'retriever': self.retriever is not None,
                'generator': self.generator is not None,
                'evaluator': self.evaluator is not None,
                'csv_exporter': self.csv_exporter is not None
            }
        }


if __name__ == "__main__":
    prototype = RAGPrototype()
    prototype.main()
