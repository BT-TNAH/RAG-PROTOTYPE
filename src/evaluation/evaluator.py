from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re
import logging

@dataclass
class F1ScoreResult:
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int

class F1ScoreEvaluator:
    """
    Évaluateur F1 Score pour comparer les sources idéales et sources trouvées par le RAG
    
    Calcule les métriques de précision, rappel et F1 score en comparant :
    - Sources idéales (colonne E du CSV) : format "fichier1_page1, fichier2_page2, ..."
    - Sources trouvées (colonne G du CSV) : format "[fichier1_page1, fichier2_page2, ...]"
    """
    
    def __init__(self):
        """
        Initialise l'évaluateur F1 Score
        """
        self.logger = logging.getLogger(__name__)
    
    def calculate_f1_score_detailed(self, ideal_sources: str, retrieved_sources: str) -> F1ScoreResult:
        """
        Calcule le score F1 entre sources idéales et sources trouvées
        
        Args:
            ideal_sources: Sources idéales de la chercheuse (colonne E)
                          Format: "fichier1_page1, fichier2_page2, ..."
            retrieved_sources: Sources trouvées par le RAG (colonne G)
                             Format: "[fichier1_page1, fichier2_page2, ...]"
            
        Returns:
            F1ScoreResult: Résultat complet de l'évaluation F1
        """
        try:
            # Parsing et normalisation des sources
            ideal_set = self.parse_sources_string(ideal_sources)
            retrieved_set = self.parse_sources_string(retrieved_sources)
            
            # Calcul des vrais positifs, faux positifs et faux négatifs
            true_positives = len(ideal_set.intersection(retrieved_set))
            false_positives = len(retrieved_set - ideal_set)
            false_negatives = len(ideal_set - retrieved_set)
            
            # Calcul de la précision
            precision = self.calculate_precision(true_positives, false_positives)
            
            # Calcul du rappel
            recall = self.calculate_recall(true_positives, false_negatives)
            
            # Calcul du F1 score
            f1_score = self.calculate_f1_from_precision_recall(precision, recall)
            
            self.logger.debug(
                f"F1 Calculation - TP: {true_positives}, FP: {false_positives}, "
                f"FN: {false_negatives}, P: {precision:.3f}, R: {recall:.3f}, F1: {f1_score:.3f}"
            )
            
            return F1ScoreResult(
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du F1 score : {e}")
            # Retour d'un résultat par défaut en cas d'erreur
            return F1ScoreResult(
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                true_positives=0,
                false_positives=0,
                false_negatives=0
            )
    
    def parse_sources_string(self, sources_string: str) -> Set[str]:
        """
        Parse une chaîne de sources et retourne un ensemble normalisé
        
        Gère les formats suivants :
        - Sources idéales : "fichier1_page1, fichier2_page2, ..."
        - Sources trouvées : "[fichier1_page1, fichier2_page2, ...]"
        - Sources vides ou invalides
        
        Args:
            sources_string: Chaîne de sources à parser
            
        Returns:
            Set[str]: Ensemble des sources normalisées
        """
        if not sources_string or not isinstance(sources_string, str):
            return set()
        
        # Nettoyage initial de la chaîne
        sources_string = sources_string.strip()
        
        if not sources_string:
            return set()
        
        try:
            # Suppression des crochets pour les sources trouvées par le RAG
            if sources_string.startswith('[') and sources_string.endswith(']'):
                sources_string = sources_string[1:-1]
            
            # Division par virgules et nettoyage
            sources_list = [source.strip() for source in sources_string.split(',')]
            
            # Normalisation de chaque source
            normalized_sources = set()
            for source in sources_list:
                if source:  # Ignorer les chaînes vides
                    normalized_source = self.normalize_source_reference(source)
                    if normalized_source:
                        normalized_sources.add(normalized_source)
            
            return normalized_sources
            
        except Exception as e:
            self.logger.warning(f"Erreur lors du parsing des sources '{sources_string}' : {e}")
            return set()
    
    def normalize_source_reference(self, source: str) -> str:
        """
        Normalise une référence source pour la comparaison
        
        Effectue les normalisations suivantes :
        1. Suppression des espaces en début/fin
        2. Conversion en minuscules pour la comparaison
        3. Normalisation du format fichier_page
        4. Gestion des variations de nommage (FR/AN, etc.)
        
        Args:
            source: Référence source à normaliser
            
        Returns:
            str: Référence normalisée
        """
        if not source or not isinstance(source, str):
            return ""
        
        # Nettoyage initial
        source = source.strip()
        
        if not source:
            return ""
        
        try:
            # Suppression des guillemets si présents
            source = source.strip('"\'')
            
            # Conversion en minuscules pour standardiser
            source = source.lower()
            
            # Normalisation des espaces multiples
            source = re.sub(r'\s+', '_', source)
            
            # Gestion spécifique des formats ELCA
            # Pattern: date_elca_nom_entretien_langue_page
            elca_pattern = r'(\d{8})_elca_([^_]+(?:_[^_]+)*)_entretien_([a-z]{2})_(\d+)'
            match = re.match(elca_pattern, source)
            
            if match:
                date, nom, langue, page = match.groups()
                # Standardisation du format
                normalized = f"{date}_elca_{nom}_entretien_{langue}_{page}"
                return normalized
            
            # Si le pattern ELCA ne correspond pas, retourner la source nettoyée
            return source
            
        except Exception as e:
            self.logger.warning(f"Erreur lors de la normalisation de la source '{source}' : {e}")
            return source.lower().strip()
    
    def evaluate_batch_questions(self, questions_data: List[Dict[str, Any]], 
                                 rag_results: List[Dict[str, Any]]) -> List[F1ScoreResult]:
        """
        Évalue un batch de questions avec le score F1
        
        Args:
            questions_data: Données des questions avec sources idéales
                          Format: [{'id': 1, 'ideal_sources': 'source1, source2', ...}, ...]
            rag_results: Résultats du RAG avec sources trouvées
                        Format: [{'id': 1, 'retrieved_sources': '[source1, source2]', ...}, ...]
            
        Returns:
            List[F1ScoreResult]: Résultats F1 pour chaque question
        """
        f1_results = []
        
        # Création d'un dictionnaire pour un accès rapide aux résultats RAG
        rag_dict = {result.get('id'): result for result in rag_results}
        
        for question_data in questions_data:
            question_id = question_data.get('id')
            ideal_sources = question_data.get('ideal_sources', '')
            
            # Récupération des sources trouvées correspondantes
            rag_result = rag_dict.get(question_id, {})
            retrieved_sources = rag_result.get('retrieved_sources', '')
            
            # Calcul du F1 score pour cette question
            f1_result = self.calculate_f1_score(ideal_sources, retrieved_sources)
            f1_results.append(f1_result)
            
            self.logger.debug(
                f"Question {question_id}: F1={f1_result.f1_score:.3f}, "
                f"P={f1_result.precision:.3f}, R={f1_result.recall:.3f}"
            )
        
        # Log des statistiques globales
        if f1_results:
            avg_f1 = sum(result.f1_score for result in f1_results) / len(f1_results)
            avg_precision = sum(result.precision for result in f1_results) / len(f1_results)
            avg_recall = sum(result.recall for result in f1_results) / len(f1_results)
            
            self.logger.info(
                f"Évaluation batch terminée - {len(f1_results)} questions - "
                f"F1 moyen: {avg_f1:.3f}, Précision moyenne: {avg_precision:.3f}, "
                f"Rappel moyen: {avg_recall:.3f}"
            )
        
        return f1_results
    
    def calculate_precision(self, true_positives: int, false_positives: int) -> float:
        """
        Calcule la précision
        
        Précision = TP / (TP + FP)
        
        Args:
            true_positives: Nombre de vrais positifs
            false_positives: Nombre de faux positifs
            
        Returns:
            float: Score de précision entre 0.0 et 1.0
        """
        if true_positives + false_positives == 0:
            # Cas où aucune source n'a été trouvée
            return 1.0 if true_positives == 0 else 0.0
        
        return true_positives / (true_positives + false_positives)
    
    def calculate_recall(self, true_positives: int, false_negatives: int) -> float:
        """
        Calcule le rappel
        
        Rappel = TP / (TP + FN)
        
        Args:
            true_positives: Nombre de vrais positifs
            false_negatives: Nombre de faux négatifs
            
        Returns:
            float: Score de rappel entre 0.0 et 1.0
        """
        if true_positives + false_negatives == 0:
            # Cas où aucune source idéale n'existe
            return 1.0 if true_positives == 0 else 0.0
        
        return true_positives / (true_positives + false_negatives)
    
    def calculate_f1_from_precision_recall(self, precision: float, recall: float) -> float:
        """
        Calcule le F1 score à partir de la précision et du rappel
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Args:
            precision: Score de précision
            recall: Score de rappel
            
        Returns:
            float: Score F1 entre 0.0 et 1.0
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_cited_sources_f1_score(self, ideal_sources: str, cited_sources: str) -> F1ScoreResult:
        """
        Calcule le score F1 entre sources idéales et sources citées dans la réponse générée
        
        Args:
            ideal_sources: Sources idéales de la chercheuse (colonne E)
            cited_sources: Sources citées dans la réponse générée (format "[source1, source2, ...]")
            
        Returns:
            F1ScoreResult: Résultat complet de l'évaluation F1 pour les sources citées
        """
        # Utilise la même logique que calculate_f1_score_detailed
        return self.calculate_f1_score_detailed(ideal_sources, cited_sources)
    
    def get_evaluation_statistics(self, f1_results: List[F1ScoreResult]) -> Dict[str, Any]:
        """
        Calcule les statistiques d'évaluation pour un ensemble de résultats F1
        
        Args:
            f1_results: Liste des résultats F1
            
        Returns:
            Dict[str, Any]: Statistiques d'évaluation
        """
        if not f1_results:
            return {
                'count': 0,
                'avg_f1': 0.0,
                'avg_precision': 0.0,
                'avg_recall': 0.0,
                'min_f1': 0.0,
                'max_f1': 0.0,
                'std_f1': 0.0,
                'total_tp': 0,
                'total_fp': 0,
                'total_fn': 0
            }
        
        f1_scores = [result.f1_score for result in f1_results]
        precisions = [result.precision for result in f1_results]
        recalls = [result.recall for result in f1_results]
        
        # Calcul de l'écart-type
        avg_f1 = sum(f1_scores) / len(f1_scores)
        variance_f1 = sum((f1 - avg_f1) ** 2 for f1 in f1_scores) / len(f1_scores)
        std_f1 = variance_f1 ** 0.5
        
        return {
            'count': len(f1_results),
            'avg_f1': avg_f1,
            'avg_precision': sum(precisions) / len(precisions),
            'avg_recall': sum(recalls) / len(recalls),
            'min_f1': min(f1_scores),
            'max_f1': max(f1_scores),
            'std_f1': std_f1,
            'total_tp': sum(result.true_positives for result in f1_results),
            'total_fp': sum(result.false_positives for result in f1_results),
            'total_fn': sum(result.false_negatives for result in f1_results)
        }


class RAGEvaluator:
    """
    Évaluateur principal pour le système RAG
    
    Intègre l'évaluateur F1 Score et d'autres métriques d'évaluation
    """
    
    def __init__(self):
        """
        Initialise l'évaluateur RAG
        """
        self.f1_evaluator = F1ScoreEvaluator()
        self.logger = logging.getLogger(__name__)

    def calculate_f1_score(self, ideal_sources: str, found_sources: List[str]) -> float:
        """
        Calcule le score F1 et retourne seulement le float
        Wrapper pour compatibilité avec evaluation_mode.py
        """
        if isinstance(found_sources, list):
            retrieved_sources = "[" + ", ".join(found_sources) + "]"
        else:
            retrieved_sources = str(found_sources)
    
        f1_result = self.f1_evaluator.calculate_f1_score_detailed(ideal_sources, retrieved_sources)
        return f1_result.f1_score
    
    def calculate_cited_sources_f1_score(self, ideal_sources: str, cited_sources: List[str]) -> float:
        """
        Calcule le score F1 pour les sources citées et retourne seulement le float
        Wrapper pour compatibilité avec evaluation_mode.py
        
        Args:
            ideal_sources: Sources idéales de la chercheuse
            cited_sources: Liste des sources citées dans la réponse générée
            
        Returns:
            float: Score F1 entre 0.0 et 1.0
        """
        if isinstance(cited_sources, list):
            cited_sources_formatted = "[" + ", ".join(cited_sources) + "]"
        else:
            cited_sources_formatted = str(cited_sources)

        f1_result = self.f1_evaluator.calculate_cited_sources_f1_score(ideal_sources, cited_sources_formatted)
        return f1_result.f1_score
    
    def evaluate_single_question(self, ideal_sources: str, retrieved_sources: str, 
                            ideal_answer: str = None, generated_answer: str = None,
                            cited_sources: str = None) -> Dict[str, Any]:  # ← Nouveau paramètre
        """
        Évalue une seule question avec toutes les métriques disponibles
        
        Args:
            ideal_sources: Sources idéales
            retrieved_sources: Sources trouvées
            ideal_answer: Réponse idéale (optionnel)
            generated_answer: Réponse générée (optionnel)
            cited_sources: Sources citées dans la réponse (optionnel, NOUVEAU)
            
        Returns:
            Dict[str, Any]: Résultats d'évaluation complets
        """
        # Évaluation F1 des sources trouvées (existant)
        f1_result = self.f1_evaluator.calculate_f1_score_detailed(ideal_sources, retrieved_sources)
        
        evaluation_result = {
            'f1_score': f1_result.f1_score,
            'precision': f1_result.precision,
            'recall': f1_result.recall,
            'true_positives': f1_result.true_positives,
            'false_positives': f1_result.false_positives,
            'false_negatives': f1_result.false_negatives
        }
        
        # NOUVEAU : Évaluation F1 des sources citées
        if cited_sources is not None:
            cited_f1_result = self.f1_evaluator.calculate_cited_sources_f1_score(ideal_sources, cited_sources)
            evaluation_result.update({
                'cited_f1_score': cited_f1_result.f1_score,
                'cited_precision': cited_f1_result.precision,
                'cited_recall': cited_f1_result.recall,
                'cited_true_positives': cited_f1_result.true_positives,
                'cited_false_positives': cited_f1_result.false_positives,
                'cited_false_negatives': cited_f1_result.false_negatives
            })
        
        return evaluation_result
    
    def evaluate_batch(self, questions_data: List[Dict[str, Any]], 
                      rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Évalue un batch de questions
        
        Args:
            questions_data: Données des questions avec sources idéales
            rag_results: Résultats du RAG
            
        Returns:
            Dict[str, Any]: Résultats d'évaluation du batch
        """
        # Évaluation F1 pour toutes les questions
        f1_results = self.f1_evaluator.evaluate_batch_questions(questions_data, rag_results)
        
        # Calcul des statistiques globales
        statistics = self.f1_evaluator.get_evaluation_statistics(f1_results)
        
        return {
            'individual_results': f1_results,
            'statistics': statistics,
            'evaluation_summary': {
                'total_questions': len(questions_data),
                'successful_evaluations': len(f1_results),
                'average_f1_score': statistics['avg_f1'],
                'evaluation_quality': self._assess_evaluation_quality(statistics['avg_f1'])
            }
        }
    
    def _assess_evaluation_quality(self, avg_f1: float) -> str:
        """
        Évalue la qualité globale basée sur le F1 score moyen
        
        Args:
            avg_f1: Score F1 moyen
            
        Returns:
            str: Évaluation qualitative
        """
        if avg_f1 >= 0.8:
            return "Excellent"
        elif avg_f1 >= 0.6:
            return "Bon"
        elif avg_f1 >= 0.4:
            return "Moyen"
        elif avg_f1 >= 0.2:
            return "Faible"
        else:
            return "Très faible"
