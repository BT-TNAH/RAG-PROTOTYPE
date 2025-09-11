from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import pandas as pd

class LLMModel(Enum):
    LLAMA31_LATEST = "llama3.1:latest"
    LLAMA33 = "llama3.3"

class EmbeddingModel(Enum):
    NOMIC_EMBED_TEXT = "nomic-embed-text:latest"
    BGE_LARGE = "bge-large:latest"
    GRANITE_EMBEDDING = "granite-embedding"
    ALL_MINILM = "all-minilm:l6-v2"

@dataclass
class RAGConfiguration:
    llm_model: LLMModel
    chunk_size: int
    chunk_overlap: int
    embedding_model: EmbeddingModel
    similarity_top_k: int

class ConfigurationManager:
    
    CHUNK_SIZES = [250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
    CHUNK_OVERLAPS = [0, 10, 20, 30, 50, 60, 70, 80, 90, 100, 800, 2000]
    SIMILARITY_TOP_K_VALUES = [3, 4, 5, 20, 30, 40, 50]
    
    def __init__(self):
        """
        Initialise le gestionnaire de configuration
        """
        pass
    
    def interactive_configuration_selection(self) -> RAGConfiguration:
        """
        Interface interactive en terminal pour sélectionner les paramètres
        
        Returns:
            RAGConfiguration: Configuration complète sélectionnée par l'utilisateur
        """
        print("\n" + "="*60)
        print("           CONFIGURATION DES PARAMÈTRES RAG")
        print("="*60)
        print("Veuillez sélectionner vos paramètres pour cette session d'évaluation.\n")
        
        try:
            # Sélection du modèle LLM
            llm_model = self._select_llm_model()
            
            # Sélection de la taille des chunks
            chunk_size = self._select_chunk_size()
            
            # Sélection du chunk overlap
            chunk_overlap = self._select_chunk_overlap(chunk_size)
            
            # Sélection du modèle d'embedding
            embedding_model = self._select_embedding_model()
            
            # Sélection du similarity top-k
            similarity_top_k = self._select_similarity_top_k()
            
            # Création de la configuration
            config = RAGConfiguration(
                llm_model=llm_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                similarity_top_k=similarity_top_k
            )
            
            # DIAGNOSTIC - Vérification configuration
            print(f"DIAGNOSTIC - similarity_top_k: {config.similarity_top_k}")
            print(f"DIAGNOSTIC - embedding_model: {config.embedding_model.value}")
            print(f"DIAGNOSTIC - chunk_size: {config.chunk_size}")

            # Validation et affichage récapitulatif
            if self.validate_configuration(config):
                self._display_configuration_summary(config)
                return config
            else:
                print("❌ Configuration invalide. Redémarrage de la sélection.")
                return self.interactive_configuration_selection()
                
        except (KeyboardInterrupt, EOFError):
            print("\n\nConfiguration annulée par l'utilisateur.")
            sys.exit(0)
        except Exception as e:
            print(f"⚠ Erreur lors de la configuration : {e}")
            print("Abandon de la configuration.")
            sys.exit(1)
    
    def get_interactive_configuration(self) -> Dict[str, any]:
        """
        Interface publique pour obtenir une configuration interactive
        Compatible avec l'appel depuis main.py
        
        Returns:
            Dict[str, any]: Configuration sous forme de dictionnaire
        """
        config = self.interactive_configuration_selection()
        return self.config_to_dict(config)
    
    def config_to_dict(self, config: RAGConfiguration) -> Dict[str, any]:
        """
        Convertit un objet RAGConfiguration en dictionnaire
        
        Args:
            config: Configuration RAG à convertir
            
        Returns:
            Dict[str, any]: Configuration sous forme de dictionnaire
        """
        return {
            'llm_model': config.llm_model.value,
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'embedding_model': config.embedding_model.value,
            'top_k': config.similarity_top_k
        }
    
    def dict_to_config(self, config_dict: Dict[str, any]) -> RAGConfiguration:
        """
        Convertit un dictionnaire en objet RAGConfiguration
        
        Args:
            config_dict: Configuration sous forme de dictionnaire
            
        Returns:
            RAGConfiguration: Objet configuration
            
        Raises:
            ValueError: Si la configuration est invalide
        """
        try:
            # Conversion des chaînes en enums
            llm_model = None
            for model in LLMModel:
                if model.value == config_dict['llm_model']:
                    llm_model = model
                    break
            
            if llm_model is None:
                raise ValueError(f"Modèle LLM non reconnu : {config_dict['llm_model']}")
            
            embedding_model = None
            for model in EmbeddingModel:
                if model.value == config_dict['embedding_model']:
                    embedding_model = model
                    break
            
            if embedding_model is None:
                raise ValueError(f"Modèle d'embedding non reconnu : {config_dict['embedding_model']}")
            
            return RAGConfiguration(
                llm_model=llm_model,
                chunk_size=config_dict['chunk_size'],
                chunk_overlap=config_dict['chunk_overlap'],
                embedding_model=embedding_model,
                similarity_top_k=config_dict['top_k']
            )
            
        except KeyError as e:
            raise ValueError(f"Clé manquante dans la configuration : {e}")
        except Exception as e:
            raise ValueError(f"Erreur lors de la conversion : {e}")
    
    def _select_llm_model(self) -> LLMModel:
        """
        Sélection interactive du modèle LLM
        
        Returns:
            LLMModel: Modèle LLM sélectionné
        """
        print("🔊 SÉLECTION DU MODÈLE LLM")
        print("-" * 30)
        print("1. llama3.1:latest (recommandé - déjà testé)")
        print("2. llama3.3 (option secondaire)")
        print()
        
        while True:
            try:
                choice = input("Votre choix (1-2) : ").strip()
                if choice == "1":
                    print("✅ Modèle sélectionné : llama3.1:latest")
                    return LLMModel.LLAMA31_LATEST
                elif choice == "2":
                    print("✅ Modèle sélectionné : llama3.3")
                    return LLMModel.LLAMA33
                else:
                    print("❌ Choix invalide. Veuillez entrer 1 ou 2.")
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                print("❌ Entrée invalide. Veuillez entrer 1 ou 2.")
    
    def _select_chunk_size(self) -> int:
        """
        Sélection interactive de la taille des chunks
        
        Returns:
            int: Taille des chunks sélectionnée
        """
        print("\n📃 SÉLECTION DE LA TAILLE DES CHUNKS")
        print("-" * 40)
        print("Petits chunks (phrase/petit paragraphe) :")
        print("  1. 250    2. 300    3. 400")
        print()
        print("Chunks moyens (paragraphe moyen/grand) :")
        print("  4. 500    5. 600")
        print()
        print("Grands chunks (page) :")
        print("  6. 700    7. 800    8. 900    9. 1000    10. 2000")
        print()
        
        while True:
            try:
                choice = input("Votre choix (1-10) : ").strip()
                if choice.isdigit() and 1 <= int(choice) <= 10:
                    chunk_size = self.CHUNK_SIZES[int(choice) - 1]
                    print(f"✅ Taille de chunk sélectionnée : {chunk_size}")
                    return chunk_size
                else:
                    print("❌ Choix invalide. Veuillez entrer un nombre entre 1 et 10.")
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                print("❌ Entrée invalide. Veuillez entrer un nombre entre 1 et 10.")
    
    def _select_chunk_overlap(self, chunk_size: int) -> int:
        """
        Sélection interactive du chunk overlap avec validation selon la taille
        
        Args:
            chunk_size: Taille des chunks pour validation
            
        Returns:
            int: Overlap sélectionné
        """
        print("\n🔄 SÉLECTION DU CHUNK OVERLAP")
        print("-" * 35)
        print("Overlap nul :")
        print("  1. 0")
        print()
        print("Petit overlap (pour phrases) :")
        print("  2. 10     3. 20     4. 30")
        print()
        print("Overlap moyen (pour paragraphes) :")
        print("  5. 50     6. 60     7. 70     8. 80     9. 90     10. 100")
        print()
        print("Grand overlap (pour pages) :")
        print("  11. 800    12. 2000")
        print()
        
        while True:
            try:
                choice = input("Votre choix (1-12) : ").strip()
                if choice.isdigit() and 1 <= int(choice) <= 12:
                    overlap = self.CHUNK_OVERLAPS[int(choice) - 1]
                    
                    # Validation que l'overlap n'excède pas la taille du chunk
                    if overlap >= chunk_size:
                        print(f"⚠️ Attention : L'overlap ({overlap}) ne peut pas être égal ou supérieur à la taille du chunk ({chunk_size}).")
                        print("Veuillez choisir un overlap plus petit.")
                        continue
                    
                    print(f"✅ Chunk overlap sélectionné : {overlap}")
                    return overlap
                else:
                    print("❌ Choix invalide. Veuillez entrer un nombre entre 1 et 12.")
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                print("❌ Entrée invalide. Veuillez entrer un nombre entre 1 et 12.")
    
    def _select_embedding_model(self) -> EmbeddingModel:
        """
        Sélection interactive du modèle d'embedding
        
        Returns:
            EmbeddingModel: Modèle d'embedding sélectionné
        """
        print("\n🧮 SÉLECTION DU MODÈLE D'EMBEDDING")
        print("-" * 40)
        print("1. nomic-embed-text:latest (recommandé - déjà testé)")
        print("2. bge-large:latest")
        print("3. granite-embedding")
        print("4. all-minilm:l6-v2")
        print()
        
        while True:
            try:
                choice = input("Votre choix (1-4) : ").strip()
                if choice == "1":
                    print("✅ Modèle d'embedding sélectionné : nomic-embed-text:latest")
                    return EmbeddingModel.NOMIC_EMBED_TEXT
                elif choice == "2":
                    print("✅ Modèle d'embedding sélectionné : bge-large:latest")
                    return EmbeddingModel.BGE_LARGE
                elif choice == "3":
                    print("✅ Modèle d'embedding sélectionné : granite-embedding")
                    return EmbeddingModel.GRANITE_EMBEDDING
                elif choice == "4":
                    print("✅ Modèle d'embedding sélectionné : all-minilm:l6-v2")
                    return EmbeddingModel.ALL_MINILM
                else:
                    print("❌ Choix invalide. Veuillez entrer un nombre entre 1 et 4.")
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                print("❌ Entrée invalide. Veuillez entrer un nombre entre 1 et 4.")
    
    def _select_similarity_top_k(self) -> int:
        """
        Sélection interactive du similarity top-k
        
        Returns:
            int: Valeur top-k sélectionnée
        """
        print("\n🔍 SÉLECTION DU SIMILARITY TOP-K")
        print("-" * 35)
        print("Petites valeurs :")
        print("  1. 3      2. 4      3. 5")
        print()
        print("Grandes valeurs :")
        print("  4. 20     5. 30     6. 40     7. 50")
        print()
        
        while True:
            try:
                choice = input("Votre choix (1-7) : ").strip()
                if choice.isdigit() and 1 <= int(choice) <= 7:
                    top_k = self.SIMILARITY_TOP_K_VALUES[int(choice) - 1]
                    print(f"✅ Similarity Top-K sélectionné : {top_k}")
                    return top_k
                else:
                    print("❌ Choix invalide. Veuillez entrer un nombre entre 1 et 7.")
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                print("❌ Entrée invalide. Veuillez entrer un nombre entre 1 et 7.")
    
    def _display_configuration_summary(self, config: RAGConfiguration) -> None:
        """
        Affiche un récapitulatif de la configuration sélectionnée
        
        Args:
            config: Configuration à afficher
        """
        print("\n" + "="*60)
        print("           RÉCAPITULATIF DE LA CONFIGURATION")
        print("="*60)
        print(f"🤖 Modèle LLM           : {config.llm_model.value}")
        print(f"📃 Taille des chunks    : {config.chunk_size}")
        print(f"🔄 Chunk overlap        : {config.chunk_overlap}")
        print(f"🧮 Modèle d'embedding   : {config.embedding_model.value}")
        print(f"🔍 Similarity Top-K     : {config.similarity_top_k}")
        print("="*60)
        
        # Demande de confirmation
        while True:
            try:
                confirm = input("\nConfirmer cette configuration ? (o/n) : ").strip().lower()
                if confirm in ['o', 'oui', 'y', 'yes']:
                    print("✅ Configuration confirmée !")
                    break
                elif confirm in ['n', 'non', 'no']:
                    print("Configuration annulée. Redémarrage de la sélection.")
                    raise ValueError("Configuration non confirmée")
                else:
                    print("Veuillez répondre par 'o' (oui) ou 'n' (non).")
            except (EOFError, KeyboardInterrupt):
                raise
    
    def validate_configuration(self, config: RAGConfiguration) -> bool:
        """
        Valide une configuration RAG
        
        Args:
            config: Configuration à valider
            
        Returns:
            bool: True si la configuration est valide
        """
        try:
            # Validation des valeurs dans les listes autorisées
            if config.chunk_size not in self.CHUNK_SIZES:
                print(f"❌ Taille de chunk invalide : {config.chunk_size}")
                return False
            
            if config.chunk_overlap not in self.CHUNK_OVERLAPS:
                print(f"❌ Chunk overlap invalide : {config.chunk_overlap}")
                return False
            
            if config.similarity_top_k not in self.SIMILARITY_TOP_K_VALUES:
                print(f"❌ Similarity Top-K invalide : {config.similarity_top_k}")
                return False
            
            # Validation que l'overlap n'excède pas la taille du chunk
            if config.chunk_overlap >= config.chunk_size:
                print(f"❌ Le chunk overlap ({config.chunk_overlap}) ne peut pas être supérieur ou égal à la taille du chunk ({config.chunk_size})")
                return False
            
            # Validation des enums
            if not isinstance(config.llm_model, LLMModel):
                print(f"❌ Modèle LLM invalide : {config.llm_model}")
                return False
            
            if not isinstance(config.embedding_model, EmbeddingModel):
                print(f"❌ Modèle d'embedding invalide : {config.embedding_model}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la validation : {e}")
            return False
    
    def get_configuration_header(self, config: RAGConfiguration) -> str:
        """
        Génère l'en-tête de colonne pour le CSV au format requis
        
        Args:
            config: Configuration RAG
            
        Returns:
            str: En-tête format "modèle_LLM_chunksize_chunkoverlap_modèle_embedding_topk"
        """
        try:
            # Extraction du nom du modèle LLM (sans les caractères spéciaux)
            llm_name = config.llm_model.value.replace(":", "_").replace(".", "_")
            
            # Extraction du nom du modèle d'embedding (sans les caractères spéciaux)
            embedding_name = config.embedding_model.value.replace(":", "_").replace(".", "_").replace("-", "_")
            
            # Construction de l'en-tête
            header = f"{llm_name}_{config.chunk_size}_{config.chunk_overlap}_{embedding_name}_{config.similarity_top_k}"
            
            return header
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération de l'en-tête : {e}")
            return "configuration_error"
    
    def get_configuration_header_from_dict(self, config_dict: Dict[str, any]) -> str:
        """
        Génère l'en-tête de colonne pour le CSV à partir d'un dictionnaire
        
        Args:
            config_dict: Configuration RAG sous forme de dictionnaire
            
        Returns:
            str: En-tête format "modèle_LLM_chunksize_chunkoverlap_modèle_embedding_topk"
        """
        try:
            # Nettoyage des noms de modèles
            llm_name = config_dict['llm_model'].replace(":", "_").replace(".", "_")
            embedding_name = config_dict['embedding_model'].replace(":", "_").replace(".", "_").replace("-", "_")
            
            # Construction de l'en-tête
            header = f"{llm_name}_{config_dict['chunk_size']}_{config_dict['chunk_overlap']}_{embedding_name}_{config_dict['top_k']}"
            
            return header
            
        except KeyError as e:
            print(f"❌ Clé manquante dans la configuration : {e}")
            return "configuration_error"
        except Exception as e:
            print(f"❌ Erreur lors de la génération de l'en-tête : {e}")
            return "configuration_error"
    
    def display_available_options(self) -> None:
        """
        Affiche toutes les options disponibles dans le terminal
        """
        print("\n" + "="*70)
        print("           OPTIONS DISPONIBLES POUR LA CONFIGURATION RAG")
        print("="*70)
        
        print("\n🤖 MODÈLES LLM DISPONIBLES :")
        print("  • llama3.1:latest (recommandé - déjà testé)")
        print("  • llama3.3 (option secondaire)")
        
        print(f"\n📃 TAILLES DE CHUNKS DISPONIBLES ({len(self.CHUNK_SIZES)} options) :")
        chunk_categories = [
            ("Petits chunks (phrase/petit paragraphe)", [250, 300, 400]),
            ("Chunks moyens (paragraphe moyen/grand)", [500, 600]),
            ("Grands chunks (page)", [700, 800, 900, 1000, 2000])
        ]
        
        for category, sizes in chunk_categories:
            print(f"  {category} :")
            sizes_str = "  ".join([f"{size:>4}" for size in sizes])
            print(f"    {sizes_str}")
        
        print(f"\n🔄 CHUNK OVERLAPS DISPONIBLES ({len(self.CHUNK_OVERLAPS)} options) :")
        overlap_categories = [
            ("Nul", [0]),
            ("Petit (pour phrases)", [10, 20, 30]),
            ("Moyen (pour paragraphes)", [50, 60, 70, 80, 90, 100]),
            ("Grand (pour pages)", [800, 2000])
        ]
        
        for category, overlaps in overlap_categories:
            print(f"  {category} :")
            overlaps_str = "  ".join([f"{overlap:>3}" for overlap in overlaps])
            print(f"    {overlaps_str}")
        
        print("\n🧮 MODÈLES D'EMBEDDING DISPONIBLES :")
        print("  • nomic-embed-text:latest (recommandé - déjà testé)")
        print("  • bge-large:latest")
        print("  • granite-embedding")
        print("  • all-minilm:l6-v2")
        
        print(f"\n🔍 SIMILARITY TOP-K DISPONIBLES ({len(self.SIMILARITY_TOP_K_VALUES)} options) :")
        topk_categories = [
            ("Petites valeurs", [3, 4, 5]),
            ("Grandes valeurs", [20, 30, 40, 50])
        ]
        
        for category, values in topk_categories:
            print(f"  {category} :")
            values_str = "  ".join([f"{value:>2}" for value in values])
            print(f"    {values_str}")
        
        print("\n" + "="*70)
        print("💡 RECOMMANDATIONS :")
        print("  • Pour des tests rapides : chunks petits (250-400) + overlap faible (0-30)")
        print("  • Pour des réponses détaillées : chunks moyens/grands (500-1000) + overlap moyen (50-100)")
        print("  • Pour une recherche précise : Top-K faible (3-5)")
        print("  • Pour une couverture large : Top-K élevé (20-50)")
        print("="*70)
    
    def create_default_configuration(self) -> RAGConfiguration:
        """
        Crée une configuration par défaut pour les tests rapides
        
        Returns:
            RAGConfiguration: Configuration par défaut
        """
        return RAGConfiguration(
            llm_model=LLMModel.LLAMA31_LATEST,
            chunk_size=500,
            chunk_overlap=50,
            embedding_model=EmbeddingModel.NOMIC_EMBED_TEXT,
            similarity_top_k=5
        )
    
    def load_configuration_from_file(self, file_path: str) -> RAGConfiguration:
        """
        Charge une configuration depuis un fichier JSON
        
        Args:
            file_path: Chemin vers le fichier de configuration
            
        Returns:
            RAGConfiguration: Configuration chargée
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si la configuration est invalide
        """
        import json
        from pathlib import Path
        
        config_file = Path(file_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé : {file_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            config = self.dict_to_config(config_dict)
            
            if not self.validate_configuration(config):
                raise ValueError("Configuration invalide dans le fichier")
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur de format JSON : {e}")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement : {e}")
    
    def save_configuration_to_file(self, config: RAGConfiguration, file_path: str) -> None:
        """
        Sauvegarde une configuration dans un fichier JSON
        
        Args:
            config: Configuration à sauvegarder
            file_path: Chemin du fichier de destination
            
        Raises:
            IOError: Si l'écriture échoue
        """
        import json
        from pathlib import Path
        
        try:
            config_dict = self.config_to_dict(config)
            
            # Ajout de métadonnées
            config_dict['metadata'] = {
                'created_at': str(pd.Timestamp.now()),
                'version': '1.0'
            }
            
            config_file = Path(file_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration sauvegardée : {file_path}")
            
        except Exception as e:
            raise IOError(f"Erreur lors de la sauvegarde : {e}")
