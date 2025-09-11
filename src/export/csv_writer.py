from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from retrieval.retriever import RetrievalResult

@dataclass
class EvaluationResult:
    """
    Structure pour stocker les résultats d'évaluation d'une question
    
    Attributes:
        question_id: Identifiant de la question (1-28)
        sources_found: Sources trouvées par le retrieval au format "[fichier_page, fichier_page, ...]"
        f1_score: Score F1 calculé (0.0 à 1.0)
        generated_response: Réponse générée par le LLM
        manual_evaluation: Colonne vide pour évaluation manuelle par la chercheuse
        cited_sources: Sources citées dans la réponse générée
        cited_f1_score: Score F1 des sources citées
    """
    question_id: int
    sources_found: str
    f1_score: float
    generated_response: str
    manual_evaluation: str = ""
    cited_sources: str = ""
    cited_f1_score: float = 0.0

class CSVExporter:
    """
    Module d'export des résultats d'évaluation - VERSION MODIFIEE
    
    CHANGEMENT MAJEUR : Préserve le fichier questions.csv original et crée un nouveau fichier de résultats
    """
    
    def __init__(self, csv_file_path: str = "src/questions.csv"):
        """
        Initialise l'exporteur CSV
        
        Args:
            csv_file_path: Chemin vers le fichier questions.csv ORIGINAL (ne sera jamais modifié)
        """
        self.original_csv_path = csv_file_path
        self.logger = logging.getLogger(__name__)
        
        # Colonnes obligatoires du fichier original
        self.required_columns = [
            'Numero_identifiant', 
            'Question', 
            'Interet_scientifique_question',
            'Typologie_question', 
            'Source_ideale_chercheuse', 
            'Reponse_ideale_chercheuse'
        ]

    def load_questions_csv(self) -> pd.DataFrame:
        """
        Charge le fichier questions.csv original (lecture seule)
        """
        try:
            self.logger.info(f"Chargement du fichier CSV original : {self.original_csv_path}")
            
            # Lecture avec paramètres stricts pour préserver la structure
            df = pd.read_csv(
                self.original_csv_path, 
                sep=',', 
                encoding='utf-8', 
                quotechar='"',
                index_col=False,
                keep_default_na=True
            )
            
            # Validation de la structure
            if not self.validate_csv_structure(df):
                raise ValueError("Structure du fichier CSV invalide")
            
            self.logger.info(f"Fichier CSV original chargé avec succès : {len(df)} questions")
            return df
            
        except FileNotFoundError:
            self.logger.error(f"Fichier non trouvé : {self.original_csv_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("Le fichier CSV est vide")
            raise ValueError("Fichier CSV vide")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du CSV : {e}")
            raise

    def create_results_filename(self, config: Dict[str, Any]) -> str:
        """
        Crée le nom du fichier de résultats basé sur la configuration
        
        Format : questions_results_MODELE_LLM_CHUNKSIZE_OVERLAP_EMBEDDING_TOPK_TIMESTAMP.csv
        
        Args:
            config: Configuration RAG utilisée
            
        Returns:
            str: Nom du fichier de résultats
        """
        try:
            # Nettoyage des noms de modèles
            llm_clean = config['llm_model'].replace(":", "_").replace(".", "_").replace("/", "_")
            embedding_clean = config['embedding_model'].replace(":", "_").replace(".", "_").replace("/", "_")
            
            # Timestamp pour éviter les conflits
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Construction du nom
            filename = (f"questions_results_{llm_clean}_{config['chunk_size']}_"
                       f"{config['chunk_overlap']}_{embedding_clean}_{config['top_k']}_{timestamp}.csv")
            
            # Chemin complet dans le même répertoire que l'original
            original_path = Path(self.original_csv_path)
            results_path = original_path.parent / filename
            
            self.logger.info(f"Nom du fichier de résultats : {results_path}")
            return str(results_path)
            
        except KeyError as e:
            self.logger.error(f"Clé de configuration manquante : {e}")
            raise ValueError(f"Configuration RAG incomplète : clé manquante {e}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du nom de fichier : {e}")
            raise ValueError(f"Impossible de créer le nom de fichier : {e}")

    def create_column_headers(self, config: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
        """
        Crée les en-têtes des 6 nouvelles colonnes selon le format spécifié
        
        Format de base : "modèle_LLM_chunksize_chunkoverlap_modèle_embedding_topk"
        Suffixes : _sources, _f1, _responses, _manual, _cited_sources, _cited_f1
        
        Args:
            config: Configuration RAG (dictionnaire)
            
        Returns:
            Tuple[str, str, str, str, str, str]: Les 6 en-têtes distincts pour les colonnes G, H, I, J, K, L
        """
        try:
            # Nettoyage des noms de modèles (suppression des caractères spéciaux)
            llm_clean = config['llm_model'].replace(":", "_").replace(".", "_").replace("/", "_")
            embedding_clean = config['embedding_model'].replace(":", "_").replace(".", "_").replace("/", "_")
            
            # Format de base : modèle_LLM_chunksize_chunkoverlap_modèle_embedding_topk
            base_header = f"{llm_clean}_{config['chunk_size']}_{config['chunk_overlap']}_{embedding_clean}_{config['top_k']}"
            
            # Création des 6 en-têtes distincts
            header_g = f"{base_header}_sources"
            header_h = f"{base_header}_f1"
            header_i = f"{base_header}_responses"
            header_j = f"{base_header}_manual"
            header_k = f"{base_header}_cited_sources"      # NOUVEAU
            header_l = f"{base_header}_cited_f1"           # NOUVEAU
            
            self.logger.info(f"En-têtes de colonnes créés : {header_g}, {header_h}, {header_i}, {header_j}, {header_k}, {header_l}")
            
            return header_g, header_h, header_i, header_j, header_k, header_l
            
        except KeyError as e:
            self.logger.error(f"Clé de configuration manquante : {e}")
            raise ValueError(f"Configuration RAG incomplète : clé manquante {e}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des en-têtes : {e}")
            raise ValueError(f"Impossible de créer les en-têtes : {e}")

    def create_results_dataframe(self, original_df: pd.DataFrame, config: Dict[str, Any], 
                                results: List[EvaluationResult]) -> pd.DataFrame:
        """
        Crée un nouveau DataFrame avec les résultats, sans modifier l'original
        
        Args:
            original_df: DataFrame original (questions.csv)
            config: Configuration RAG utilisée
            results: Résultats de l'évaluation
            
        Returns:
            pd.DataFrame: Nouveau DataFrame avec les colonnes de résultats ajoutées
        """
        try:
            self.logger.info("Création du DataFrame de résultats")
            
            # 1. Copie complète du DataFrame original pour préserver la structure
            results_df = original_df.copy(deep=True)
            
            # 2. Création des en-têtes
            header_g, header_h, header_i, header_j, header_k, header_l = self.create_column_headers(config)
            
            # 3. Création d'un dictionnaire de mapping par question_id
            results_dict = {}
            for result in results:
                try:
                    if hasattr(result, 'question_id'):
                        question_id = int(float(str(result.question_id).strip()))  # ← NOUVELLE LIGNE
                    elif isinstance(result, dict):
                        question_id = int(float(str(result['question_id']).strip()))  # ← NOUVELLE LIGNE
                    else:
                        self.logger.warning(f"Résultat sans question_id valide: {result}")
                        continue
                    
                    results_dict[question_id] = result
                except (ValueError, TypeError, KeyError) as e:
                    self.logger.warning(f"Impossible d'extraire question_id de {result}: {e}")
                    continue
            
            # 4. Création des données dans l'ordre exact du DataFrame
            sources_data = []
            f1_scores_data = []
            responses_data = []
            manual_eval_data = []
            cited_sources_data = []
            cited_f1_scores_data = []
            
            # 5. Parcourir le DataFrame dans l'ordre pour garantir l'alignement
            for idx, row in results_df.iterrows():
                try:
                    question_id = int(float(str(row['Numero_identifiant']).replace(',', '.')))
                except (ValueError, TypeError):
                    self.logger.warning(f"ID invalide à l'index {idx}: {row['Numero_identifiant']}")
                    question_id = idx + 1  # Fallback
                
                if question_id in results_dict:
                    result = results_dict[question_id]
                    
                    # Extraction sécurisée des données
                    if hasattr(result, 'sources_found'):
                        sources_data.append(self._format_sources_for_csv(result.sources_found))
                        f1_scores_data.append(result.f1_score)
                        responses_data.append(result.generated_response)
                        manual_eval_data.append(result.manual_evaluation)
                        cited_sources_data.append(getattr(result, 'cited_sources', '[]'))
                        cited_f1_scores_data.append(getattr(result, 'cited_f1_score', 0.0))
                    elif isinstance(result, dict):
                        sources_data.append(self._format_sources_for_csv(result.get('sources_found_formatted', '[]')))
                        f1_scores_data.append(result.get('f1_score', 0.0))
                        responses_data.append(result.get('generated_response', 'ERREUR: Réponse manquante'))
                        manual_eval_data.append(result.get('manual_evaluation', ''))
                        cited_sources_data.append(result.get('cited_sources', '[]'))
                        cited_f1_scores_data.append(result.get('cited_f1_score', 0.0))
                else:
                    # Valeurs par défaut pour question manquante
                    self.logger.warning(f"Résultat manquant pour question_id {question_id}")
                    sources_data.append("[]")
                    f1_scores_data.append(0.0)
                    responses_data.append("ERREUR: Résultat manquant")
                    manual_eval_data.append("")
                    cited_sources_data.append("[]")
                    cited_f1_scores_data.append(0.0)
            
            # 6. Vérification de cohérence
            expected_length = len(results_df)
            data_lengths = [len(sources_data), len(f1_scores_data), len(responses_data), 
                           len(manual_eval_data), len(cited_sources_data), len(cited_f1_scores_data)]
            
            if not all(length == expected_length for length in data_lengths):
                raise ValueError(f"Longueurs incohérentes: DataFrame={expected_length}, Données={data_lengths}")
            
            # 7. Ajout des nouvelles colonnes
            results_df[header_g] = sources_data
            results_df[header_h] = f1_scores_data
            results_df[header_i] = responses_data
            results_df[header_j] = manual_eval_data
            results_df[header_k] = cited_sources_data
            results_df[header_l] = cited_f1_scores_data
            
            self.logger.info(f"DataFrame de résultats créé avec 6 nouvelles colonnes")
            return results_df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du DataFrame de résultats : {e}")
            raise

    def save_results_csv(self, results_df: pd.DataFrame, results_path: str) -> None:
        """
        Sauvegarde le DataFrame de résultats dans un nouveau fichier
        
        Args:
            results_df: DataFrame avec les résultats
            results_path: Chemin du fichier de résultats
        """
        try:
            self.logger.info(f"Sauvegarde du fichier de résultats : {results_path}")
            
            # Sauvegarde avec paramètres stricts pour préserver la structure
            results_df.to_csv(
                results_path, 
                sep=',', 
                encoding='utf-8', 
                quotechar='"',
                index=False,
                float_format='%.1f'  # Format des décimales pour les IDs
            )
            
            self.logger.info("Fichier de résultats sauvegardé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde : {e}")
            raise IOError(f"Impossible de sauvegarder le fichier de résultats : {e}")

    def _format_sources_for_csv(self, sources_found: str) -> str:
        """
        Formate les sources pour l'export CSV
        """
        try:
            if isinstance(sources_found, str):
                if sources_found.startswith('[') and sources_found.endswith(']'):
                    sources_content = sources_found[1:-1]  # Enlever les crochets
                    if sources_content:
                        sources_list = [s.strip() for s in sources_content.split(',') if s.strip()]
                        
                        validated_sources = []
                        for source in sources_list:
                            if '_meta' in source or '_' in source:
                                validated_sources.append(source)
                            else:
                                self.logger.warning(f"Source sans format attendu : {source}")
                                validated_sources.append(source)
                        
                        if validated_sources:
                            formatted = f"[{', '.join(validated_sources)}]"
                            return formatted
                
                return sources_found
            elif isinstance(sources_found, list):
                if sources_found:
                    return f"[{', '.join(str(s) for s in sources_found)}]"
                else:
                    return "[]"
            else:
                return str(sources_found) if sources_found else "[]"
                
        except Exception as e:
            self.logger.error(f"Erreur lors du formatage des sources : {e}")
            return str(sources_found) if sources_found else "[]"

    def validate_csv_structure(self, df: pd.DataFrame) -> bool:
        """
        Valide la structure du CSV (28 questions, colonnes A-F présentes)
        """
        try:
            self.logger.info("Validation de la structure du fichier CSV")
            
            # Vérification des colonnes obligatoires
            missing_columns = []
            for col in self.required_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                self.logger.error(f"Colonnes manquantes : {missing_columns}")
                return False
            
            # Vérification du nombre de lignes
            if len(df) < 28:
                self.logger.error(f"Nombre de questions insuffisant : {len(df)} (minimum : 28)")
                return False
            
            self.logger.info("Structure du fichier CSV validée avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation : {e}")
            return False

    def export_evaluation_results(self, config: Dict[str, Any], 
                                 results: List[EvaluationResult]) -> str:
        """
        Export vers un nouveau fichier sans modifier l'original
        
        Args:
            config: Configuration RAG utilisée
            results: Résultats de l'évaluation
            
        Returns:
            str: Chemin du fichier de résultats créé
        """
        try:
            self.logger.info("Début de l'export des résultats d'évaluation (nouveau fichier)")
            
            # 1. Chargement du fichier original (lecture seule)
            original_df = self.load_questions_csv()
            
            # 2. Validation et nettoyage
            if not self.validate_csv_structure(original_df):
                raise ValueError("Structure du fichier CSV original invalide")
            
            # 3. Nettoyage du DataFrame original
            original_df_clean = original_df.dropna(how='all').reset_index(drop=True)
            
            # 4. Vérification de cohérence
            expected_count = len(original_df_clean)
            actual_results = len(results)
            
            self.logger.info(f"Questions dans le fichier original: {expected_count}")
            self.logger.info(f"Résultats à traiter: {actual_results}")
            
            if actual_results != expected_count:
                self.logger.warning(f"Nombre de résultats ({actual_results}) à questions ({expected_count})")
                
                # Filtrage des résultats pour correspondre aux IDs valides
                valid_ids = set()
                for idx, row in original_df_clean.iterrows():
                    try:
                        question_id = int(float(str(row['Numero_identifiant']).replace(',', '.')))
                        valid_ids.add(question_id)
                    except (ValueError, TypeError):
                        self.logger.warning(f"ID invalide: {row['Numero_identifiant']}")
                        valid_ids.add(idx + 1)  # Fallback
                
                valid_results = []
                for result in results:
                    try:
                        result_id = int(result.question_id) if hasattr(result, 'question_id') else int(result['question_id'])
                        if result_id in valid_ids:
                            valid_results.append(result)
                    except (ValueError, TypeError, KeyError):
                        self.logger.warning(f"Résultat ignoré (ID invalide): {result}")
                        continue
                
                results = valid_results
                self.logger.info(f"Résultats filtrés: {len(results)}")
            
            # 5. Création du nom du fichier de résultats
            results_file_path = self.create_results_filename(config)
            
            # 6. Création du DataFrame de résultats
            results_df = self.create_results_dataframe(original_df_clean, config, results)
            
            # 7. Sauvegarde du nouveau fichier
            self.save_results_csv(results_df, results_file_path)
            
            self.logger.info(f"Export terminé avec succès : {results_file_path}")
            print(f"\nâœ… Fichier original préservé : {self.original_csv_path}")
            print(f"ðŸ“Š Résultats exportés vers : {results_file_path}")
            
            return results_file_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export : {e}")
            raise

    def export_results(self, results: List[Dict[str, Any]], questions_file_path: str, config: Dict[str, Any]) -> str:
        """
        Méthode appelée par evaluation_mode.py - convertit et redirige vers export_evaluation_results
        """
        try:
            # Conversion des résultats vers le format EvaluationResult
            evaluation_results = []
            for result in results:
                eval_result = EvaluationResult(
                    question_id=result['question_id'],
                    sources_found=str(result['sources_found']),
                    f1_score=result['f1_score'],
                    generated_response=result['generated_response'],
                    manual_evaluation=result.get('manual_evaluation', ""),
                    cited_sources=result.get('cited_sources', '[]'),
                    cited_f1_score=result.get('cited_f1_score', 0.0)
                )
                evaluation_results.append(eval_result)
            
            # Redirection vers la méthode principale
            return self.export_evaluation_results(config, evaluation_results)
            
        except Exception as e:
            self.logger.error(f"Erreur dans export_results : {e}")
            raise
