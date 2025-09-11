from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import logging
from pathlib import Path
import time
from datetime import datetime

# Imports cohérents avec main.py
from retrieval.retriever import DocumentRetriever
try:
    from retrieval.retriever import RetrievalResult
except ImportError:
    class RetrievalResult:
        def __init__(self, chunk, similarity_score, source_reference):
            self.chunk = chunk
            self.similarity_score = similarity_score
            self.source_reference = source_reference
from generation.generator import LLMGenerator as ResponseGenerator
from evaluation.evaluator import RAGEvaluator
from export.csv_writer import CSVExporter, EvaluationResult


class EvaluationMode:
    """
    Mode évaluation pour le traitement automatisé des 28 questions de test
    """
    
    def __init__(self, retriever: Optional[DocumentRetriever] = None,
                 generator: Optional[ResponseGenerator] = None,
                 evaluator: Optional[RAGEvaluator] = None,
                 csv_exporter: Optional[CSVExporter] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialise le mode évaluation
        """
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator
        self.csv_exporter = csv_exporter
        self.current_config = config
        
        # Configuration du logging
        self.logger = logging.getLogger(__name__)
        
        # Statistiques de session
        self.session_stats = {
            'start_time': None,
            'end_time': None,
            'total_questions': 0,
            'processed_questions': 0,
            'average_f1_score': 0.0,
            'processing_errors': 0
        }
    
    def set_components(self, retriever: DocumentRetriever = None,
                      generator: ResponseGenerator = None,
                      evaluator: RAGEvaluator = None,
                      csv_exporter: CSVExporter = None,
                      config: Dict[str, Any] = None) -> None:
        """
        Méthode pour assigner les composants après initialisation
        """
        if retriever:
            self.retriever = retriever
        if generator:
            self.generator = generator
        if evaluator:
            self.evaluator = evaluator
        if csv_exporter:
            self.csv_exporter = csv_exporter
        if config:
            self.current_config = config
    
    def run_evaluation(self, questions_file_path: str) -> None:
        """
        Lance l'évaluation sur les 28 questions de test
        """
        try:
            self.session_stats['start_time'] = datetime.now()
            
            print("\n🔬 MODE ÉVALUATION - 28 QUESTIONS DE TEST")
            print("=" * 60)
            
            # Vérification des composants
            if not self._components_ready():
                raise Exception("Composants RAG non initialisés")
            
            print(f"📚 Traitement des questions depuis {questions_file_path}...")

            # Traitement des questions
            results = self.process_all_questions(questions_file_path)

            # Export des résultats
            print("\n💾 Export des résultats...")
            
            # Export des résultats
            results_file_path = self.csv_exporter.export_results(
                results, 
                "",  # Paramètre non utilisé mais requis par la signature
                self.current_config
            )
            
            # Statistiques finales
            self.session_stats['end_time'] = datetime.now()
            self.display_final_statistics(results)
            
            print(f"\n✅ Mode évaluation terminé avec succès !")
            print(f"\nRésultats exportés dans : {results_file_path}")
            
        except Exception as e:
            self.logger.error(f"Erreur dans run_evaluation: {e}")
            print(f"❌ Erreur lors de l'évaluation : {e}")
            raise
    
    def process_all_questions(self, questions_file_path: str) -> List[Dict[str, Any]]:
        """
        Traite séquentiellement les 28 questions
        """
        try:
            # Chargement du fichier questions.csv
            df = pd.read_csv(questions_file_path, sep=',', encoding='utf-8', quotechar='"')
            df = df.dropna(subset=['Question'], how='any')  # Filtre les lignes vides
            
            # CORRECTION : Affichage des colonnes disponibles pour debug
            self.logger.info(f"Colonnes disponibles dans CSV : {list(df.columns)}")
            
            # CORRECTION : Mapping flexible des colonnes
            column_mappings = [
                # Essayer différentes variantes de noms de colonnes
                {
                    'numero': ['Numero_identifiant', 'numero', 'id', 'Numero'],
                    'question': ['Question', 'question', 'Query'],
                    'interet_scientifique': ['Interet_scientifique_question', 'interet', 'scientific_interest'],
                    'typologie': ['Typologie_question', 'typologie', 'type'],
                    'source_ideale': ['Source_ideale_chercheuse', 'source_ideale', 'expected_source'],
                    'reponse_ideale': ['Reponse_ideale_chercheuse', 'reponse_ideale', 'expected_answer']
                }
            ]
            
            # Trouver le bon mapping
            actual_mapping = {}
            for key, possible_names in column_mappings[0].items():
                for name in possible_names:
                    if name in df.columns:
                        actual_mapping[key] = name
                        break
                if key not in actual_mapping:
                    self.logger.error(f"Impossible de trouver une colonne pour '{key}' parmi {df.columns}")
                    raise Exception(f"Colonne manquante pour '{key}'")
            
            self.logger.info(f"Mapping des colonnes utilisé : {actual_mapping}")
            
            self.session_stats['total_questions'] = len(df)
            results = []
            
            print(f"\n📄 Traitement de {len(df)} questions...")
            
            # Traitement séquentiel avec validation des données
            for index, row in df.iterrows():
                try:
                    self.display_progress(index + 1, len(df))
                    
                    # CORRECTION : Validation des valeurs NaN
                    question_data = {}
                    for key, col_name in actual_mapping.items():
                        value = row[col_name]
                        # Gérer les valeurs NaN
                        if pd.isna(value) or str(value).lower() == 'nan':
                            if key == 'question':
                                self.logger.error(f"Question vide à l'index {index}, ignorée")
                                continue  # Passer à la question suivante
                            else:
                                question_data[key] = ''  # Valeur par défaut
                        else:
                            question_data[key] = str(value).strip()
                    
                    # Vérifier que nous avons au minimum une question
                    if 'question' not in question_data or not question_data['question']:
                        self.logger.warning(f"Question vide à l'index {index}, création d'un résultat d'erreur")
                        error_result = {
                            'question_id': f'empty_{index}',
                            'question': 'Question vide ou NaN',
                            'sources_found': [],
                            'sources_found_formatted': "[]",
                            'f1_score': 0.0,
                            'generated_response': "ERREUR: Question vide",
                            'manual_evaluation': "",
                            'processing_time': 0.0,
                            'cited_sources': "[]",
                            'cited_f1_score': 0.0
                        }
                        results.append(error_result)
                        continue
                    
                    result = self.process_single_question(question_data)
                    
                    # Validation du résultat
                    if result is None:
                        self.logger.error(f"process_single_question a retourné None pour la question {index}")
                        result = {
                            'question_id': question_data.get('numero', f'error_{index}'),
                            'question': question_data.get('question', 'Question non disponible'),
                            'sources_found': [],
                            'sources_found_formatted': "[]",
                            'f1_score': 0.0,
                            'generated_response': "ERREUR: Résultat None retourné",
                            'manual_evaluation': "",
                            'processing_time': 0.0,
                            'cited_sources': "[]",
                            'cited_f1_score': 0.0
                        }
                    
                    results.append(result)
                    self.session_stats['processed_questions'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Erreur question {index}: {e}")
                    self.session_stats['processing_errors'] += 1
                    
                    error_result = {
                        'question_id': f'error_{index}',
                        'question': 'Erreur de traitement',
                        'sources_found': [],
                        'sources_found_formatted': "[]",
                        'f1_score': 0.0,
                        'generated_response': f"ERREUR: {str(e)}",
                        'manual_evaluation': "",
                        'processing_time': 0.0,
                        'cited_sources': "[]",
                        'cited_f1_score': 0.0
                    }
                    results.append(error_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur critique dans process_all_questions: {e}")
            raise
    
    def process_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite une question individuelle
        """
        start_time = time.time()

        # Validation des types d'entrée
        if not isinstance(question_data, dict):
            question_data = {'question': str(question_data), 'numero': 'unknown'}
            
        for key in ['question', 'numero']:
            if key in question_data and not isinstance(question_data[key], str):
                question_data[key] = str(question_data[key])
        
        try:
            # Vérification de l'état du retriever
            if not self.retriever or not hasattr(self.retriever, 'vector_store'):
                self.logger.error("Retriever mal initialisé")
                processing_time = time.time() - start_time
                return {
                    'question_id': question_data.get('numero', 'unknown'),
                    'question': question_data.get('question', 'Question non disponible'),
                    'sources_found': [],
                    'sources_found_formatted': "[]",
                    'f1_score': 0.0,
                    'generated_response': "ERREUR: Retriever non initialisé",
                    'manual_evaluation': "",
                    'processing_time': processing_time,
                    'cited_sources': "[]",  # Format cohérent
                    'cited_f1_score': 0.0
                }

            # 1. Retrieval des chunks pertinents avec gestion d'erreur spécifique
            retrieval_results = []
            try:
                # Vérification préalable des composants
                if not hasattr(self.retriever, 'retrieve'):
                    raise AttributeError("Méthode retrieve() manquante dans le retriever")
                
                # Validation de la question
                question_text = question_data.get('question', '').strip()
                if not question_text:
                    raise ValueError("Question vide pour le retrieval")
                
                self.logger.debug(f"Début retrieval pour: '{question_text[:50]}...'")
                retrieval_results = self.retriever.retrieve(question_text)
                self.logger.debug(f"Retrieval terminé: {len(retrieval_results)} résultats")
                
            except Exception as retrieval_error:
                self.logger.error(f"Erreur dans le retrieval: {retrieval_error}")
                
                # Fallback: essayer l'accès direct au vector store
                try:
                    self.logger.info("Tentative de retrieval direct via vector store")
                    query_embedding = self.retriever.embedding_processor.generate_query_embedding(question_data['question'])
                    
                    # Essayer différentes méthodes de recherche
                    if hasattr(self.retriever.vector_store, 'search_similar'):
                        vector_results = self.retriever.vector_store.search_similar(query_embedding, self.retriever.top_k)
                    elif hasattr(self.retriever.vector_store, 'search'):
                        vector_results = self.retriever.vector_store.search(query_embedding, self.retriever.top_k)
                    else:
                        raise AttributeError("Aucune méthode de recherche disponible")
                    
                    # Conversion des résultats
                    retrieval_results = []
                    for embedded_chunk, score in vector_results:
                        # Extraire le chunk original
                        actual_chunk = embedded_chunk.chunk if hasattr(embedded_chunk, 'chunk') else embedded_chunk
                        
                        # Créer RetrievalResult
                        source_ref = self.retriever.format_source_reference(actual_chunk)
                        result = RetrievalResult(
                            chunk=actual_chunk,
                            similarity_score=score,
                            source_reference=source_ref
                        )
                        retrieval_results.append(result)
                    
                    self.logger.info(f"Retrieval de secours réussi: {len(retrieval_results)} résultats")
                    
                except Exception as fallback_error:
                    self.logger.error(f"Échec du retrieval de secours: {fallback_error}")
                    retrieval_results = []
            
            # 2. Formatage des sources trouvées
            sources_found = []
            chunks_for_generation = []
            
            for result in retrieval_results:
                try:
                    if hasattr(result, 'source_reference') and hasattr(result, 'chunk'):
                        # RetrievalResult valide
                        sources_found.append(result.source_reference)
                        chunks_for_generation.append(result.chunk)
                    else:
                        # Fallback pour autres types
                        source_ref = "source_unknown"
                        if hasattr(result, 'metadata') and result.metadata:
                            metadata = result.metadata
                            filename = metadata.get('filename', 'unknown').replace('.pdf', '')
                            page_num = metadata.get('page_number', 1)
                            source_ref = f"{filename}_{page_num}"
                        
                        sources_found.append(source_ref)
                        
                        # Extraire le chunk selon le type
                        if hasattr(result, 'content'):
                            chunks_for_generation.append(result)
                        elif hasattr(result, 'text'):
                            # Créer un objet similaire à DocumentChunk
                            chunk_like = type('ChunkLike', (), {
                                'content': result.text,
                                'metadata': getattr(result, 'metadata', {})
                            })()
                            chunks_for_generation.append(chunk_like)
                        else:
                            chunks_for_generation.append(result)
                except Exception as e:
                    self.logger.warning(f"Erreur traitement résultat retrieval : {e}")
                    continue  
            
            # Formatage pour CSV
            sources_formatted = self._format_sources_for_csv(sources_found)
            
            # 3. Génération de la réponse
            generated_response = self.generator.generate_response(
                question_data['question'], 
                chunks_for_generation,
                retrieval_results=retrieval_results
            )

            # Extraire le texte de l'objet GenerationResponse
            response_text = generated_response  # Puisque generate_response() retourne maintenant directement une string

            # 4. Évaluation F1
            f1_score = 0.0
            if self.evaluator and sources_found:
                try:
                    f1_score = self.evaluator.calculate_f1_score(
                        question_data['source_ideale'],
                        sources_found
                    )
                except Exception as eval_error:
                    self.logger.warning(f"Erreur lors de l'évaluation F1 : {eval_error}")
                    f1_score = 0.0

            # 5. Extraction des sources citées dans la réponse générée
            cited_sources = []
            cited_f1_score = 0.0
            cited_sources_formatted = "[]"  # Initialisation par défaut
            
            if hasattr(self.generator, 'get_last_cited_sources'):
                try:
                    cited_sources = self.generator.get_last_cited_sources()
                    cited_sources_formatted = self.generator.format_cited_sources_for_csv(cited_sources) if cited_sources else "[]"
                    self.logger.debug(f"Sources citées extraites : {cited_sources}")
                    
                    # Calcul du F1 score pour les sources citées (même si vides)
                    if self.evaluator:
                        cited_f1_score = self.evaluator.calculate_cited_sources_f1_score(
                            question_data['source_ideale'],
                            cited_sources
                        )
                    else:
                        cited_f1_score = 0.0
                        
                except Exception as cited_error:
                    self.logger.warning(f"Erreur lors de l'extraction des sources citées : {cited_error}")
                    cited_sources = []
                    cited_f1_score = 0.0
            else:
                self.logger.warning("Méthode get_last_cited_sources() non disponible dans le générateur")
                cited_sources = []
                cited_f1_score = 0.0

            # 6. Création du résultat
            processing_time = time.time() - start_time

            result = {
                'question_id': question_data['numero'],
                'question': question_data['question'],
                'sources_found': sources_found,
                'sources_found_formatted': sources_formatted,
                'f1_score': f1_score,
                'generated_response': response_text,
                'manual_evaluation': "",
                'processing_time': processing_time,
                'cited_sources': cited_sources_formatted,  # Format cohérent
                'cited_f1_score': cited_f1_score
            }

            # Return explicite du résultat
            self.logger.debug(f"Résultat créé avec succès pour la question {result['question_id']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur critique dans process_single_question: {e}")
            processing_time = time.time() - start_time if 'start_time' in locals() else 0.0
    
            # Résultat d'erreur avec vérifications robustes
            try:
                question_id = question_data.get('numero', 'unknown') if question_data else 'unknown'
                question_text = question_data.get('question', 'Question non disponible') if question_data else 'Question non disponible'
            except Exception:
                question_id = 'error_unknown'
                question_text = 'Erreur de traitement'
    
            # Retour garanti d'un dictionnaire valide
            error_result = {
                'question_id': question_id,
                'question': question_text,
                'sources_found': [],
                'sources_found_formatted': "[]",
                'f1_score': 0.0,
                'generated_response': f"ERREUR: {str(e)}",
                'manual_evaluation': "",
                'processing_time': processing_time,
                'cited_sources': "[]",  # Format cohérent
                'cited_f1_score': 0.0
            }
    
            self.logger.info(f"Retour d'un résultat d'erreur valide: {error_result['question_id']}")
            return error_result
    
    def _format_sources_for_csv(self, sources: List[str]) -> str:
        """
        Formate la liste des sources pour le CSV
        """
        if not sources:
            return "[]"
        
        try:
            # Suppression des doublons tout en préservant l'ordre
            unique_sources = []
            seen = set()
            for source in sources:
                if source not in seen:
                    unique_sources.append(source)
                    seen.add(source)
            
            # Formatage de la liste
            sources_str = ", ".join(unique_sources)
            return f"[{sources_str}]"
            
        except Exception as e:
            self.logger.error(f"Erreur lors du formatage des sources : {e}")
            return "[]"
    
    def display_progress(self, current: int, total: int) -> None:
        """
        Affiche la progression du traitement
        """
        percentage = (current / total) * 100
        progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
        
        print(f"\r  [{progress_bar}] Question {current:2d}/{total} ({percentage:5.1f}%)", end="", flush=True)
        
        if current == total:
            print()  # Nouvelle ligne à la fin
    
    def display_final_statistics(self, results: List[Dict[str, Any]]) -> None:
        """
        Affiche les statistiques finales de l'évaluation
        """
        print("\n" + "=" * 60)
        print("           STATISTIQUES FINALES DE L'ÉVALUATION")
        print("=" * 60)
    
        # Calculs statistiques avec gestion du type GenerationResponse
        total_questions = len(results)
    
        # CORRECTION: Gestion du type GenerationResponse
        successful_questions = 0
        for r in results:
            generated_response = r['generated_response']
        
            # Extraire le texte selon le type
            if hasattr(generated_response, 'response'):
                response_text = generated_response.response
            elif hasattr(generated_response, 'text'):
                response_text = generated_response.text
            elif isinstance(generated_response, str):
                response_text = generated_response
            else:
                response_text = str(generated_response)
        
            # Vérifier si ce n'est pas une erreur
            if not response_text.startswith("ERREUR"):
                successful_questions += 1
    
        f1_scores = [r['f1_score'] for r in results if r['f1_score'] > 0]
        avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
        processing_times = [r['processing_time'] for r in results]
        total_time = self.session_stats['end_time'] - self.session_stats['start_time']
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
    
        # Affichage des statistiques
        print(f"📊 Questions traitées      : {successful_questions}/{total_questions}")
        print(f"⚠️ Erreurs rencontrées    : {self.session_stats['processing_errors']}")
        print(f"🎯 Score F1 moyen         : {avg_f1_score:.3f}")
        print(f"⏱️ Temps total           : {total_time}")
        print(f"⚡ Temps moyen/question   : {avg_processing_time:.2f}s")

        # NOUVEAU : Statistiques des sources citées
        cited_f1_scores = [r.get('cited_f1_score', 0.0) for r in results if r.get('cited_f1_score', 0.0) > 0]
        avg_cited_f1_score = sum(cited_f1_scores) / len(cited_f1_scores) if cited_f1_scores else 0.0
        
        print(f"🎯 Score F1 sources citées : {avg_cited_f1_score:.3f}")
    
        # Configuration utilisée
        if self.current_config:
            print(f"\n🔧 Configuration utilisée :")
            print(f"   • Modèle LLM          : {self.current_config.get('llm_model', 'Non défini')}")
            print(f"   • Chunk Size          : {self.current_config.get('chunk_size', 'Non défini')}")
            print(f"   • Chunk Overlap       : {self.current_config.get('chunk_overlap', 'Non défini')}")
            print(f"   • Modèle Embedding    : {self.current_config.get('embedding_model', 'Non défini')}")
            print(f"   • Top-K               : {self.current_config.get('top_k', 'Non défini')}")
    
        # Répartition des scores F1
        if f1_scores:
            score_ranges = {
                "Excellent (≥ 0.8)": len([s for s in f1_scores if s >= 0.8]),
                "Bon (0.6-0.8)": len([s for s in f1_scores if 0.6 <= s < 0.8]),
                "Moyen (0.4-0.6)": len([s for s in f1_scores if 0.4 <= s < 0.6]),
                "Faible (< 0.4)": len([s for s in f1_scores if s < 0.4])
            }
        
            print(f"\n📈 Répartition des scores F1 :")
            for range_name, count in score_ranges.items():
                percentage = (count / len(f1_scores)) * 100
                print(f"   • {range_name:<20} : {count:2d} questions ({percentage:4.1f}%)")

        # NOUVEAU : Répartition des scores F1 des sources citées
        if cited_f1_scores:
            cited_score_ranges = {
                "Excellent (≥ 0.8)": len([s for s in cited_f1_scores if s >= 0.8]),
                "Bon (0.6-0.8)": len([s for s in cited_f1_scores if 0.6 <= s < 0.8]),
                "Moyen (0.4-0.6)": len([s for s in cited_f1_scores if 0.4 <= s < 0.6]),
                "Faible (< 0.4)": len([s for s in cited_f1_scores if s < 0.4])
            }
            
            print(f"\n📈 Répartition des scores F1 sources citées :")
            for range_name, count in cited_score_ranges.items():
                percentage = (count / len(cited_f1_scores)) * 100
                print(f"   • {range_name:<20} : {count:2d} questions ({percentage:4.1f}%)")
    
        # Mise à jour des statistiques de session
        self.session_stats['average_f1_score'] = avg_f1_score
    
        print("=" * 60)
    
    def _components_ready(self) -> bool:
        """
        Vérifie si les composants RAG sont prêts avec diagnostic détaillé
        """
        components_status = {
            'retriever': self.retriever is not None,
            'generator': self.generator is not None,
            'csv_exporter': self.csv_exporter is not None,
            'config': self.current_config is not None
        }
    
        # Diagnostic approfondi pour le retriever
        if self.retriever is not None:
            # Vérification de la compatibilité du vector store
            if hasattr(self.retriever, 'vector_store'):
                vs = self.retriever.vector_store
                if hasattr(vs, 'search_similar'):
                    self.logger.info("VectorStore compatible (search_similar)")
                elif hasattr(vs, 'search'):
                    self.logger.warning("ATTENTION: VectorStore utilise search() au lieu de search_similar()")
                else:
                    self.logger.error("ERREUR: VectorStore incompatible - méthodes de recherche manquantes")
                    components_status['retriever'] = False
    
        # Log des composants manquants pour debug
        missing_components = [name for name, status in components_status.items() if not status]
        if missing_components:
            self.logger.error(f"Composants manquants : {missing_components}")
    
        return all(components_status.values())
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de la session d'évaluation
        """
        return self.session_stats.copy()
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration actuellement utilisée
        """
        return self.current_config if self.current_config else {}
