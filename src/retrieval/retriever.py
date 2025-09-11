from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from storage.vector_store import FaissVectorStore
from processing.embedding import EmbeddingProcessor
from processing.chunking import DocumentChunk

@dataclass
class RetrievalResult:
    """
    Résultat de recherche contenant un chunk avec ses métadonnées
    
    Attributes:
        chunk: Chunk de document récupéré
        similarity_score: Score de similarité avec la requête
        source_reference: Référence formatée "fichier_page"
    """
    chunk: DocumentChunk
    similarity_score: float
    source_reference: str  # Format "fichier_page"

class DocumentRetriever:
    """
    Module de retrieval pour la recherche de chunks pertinents
    
    Utilise Faiss Vector Store pour la recherche par similarité sémantique
    et gère le formatage des références sources selon les spécifications.
    """
    
    def __init__(self, vector_store: FaissVectorStore, embedding_processor: EmbeddingProcessor, top_k: int):
        """
        Initialise le retriever avec ses composants
        
        Args:
            vector_store: Store vectoriel pour la recherche
            embedding_processor: Processeur d'embeddings pour les requêtes
            top_k: Nombre de chunks à récupérer
        """
        self.vector_store = vector_store
        self.embedding_processor = embedding_processor
        self.top_k = top_k
        self.logger = logging.getLogger(__name__)
        
        # Validation des paramètres
        if self.top_k <= 0:
            raise ValueError("top_k doit être supérieur à 0")
        if self.top_k > 100:
            self.logger.warning(f"top_k très élevé ({self.top_k}), cela peut impacter les performances")
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        try:
            # Validation du type de query
            if not isinstance(query, str):
                query = str(query)  # Convertir en string si ce n'est pas déjà le cas
            
            return self.retrieve_relevant_chunks(query)
        except Exception as e:
            self.logger.error(f"Erreur dans retrieve() : {e}")
            raise RuntimeError(f"Erreur lors de la recherche de chunks : {e}")
    
    def retrieve_relevant_chunks(self, query: str) -> List[RetrievalResult]:
        """
        Récupère les chunks pertinents pour une requête
        
        Processus :
        1. Génération de l'embedding de la requête
        2. Recherche par similarité dans le vector store
        3. Formatage des résultats avec métadonnées
        4. Tri par score de similarité décroissant
        
        Args:
            query: Requête utilisateur
            
        Returns:
            List[RetrievalResult]: Liste des chunks pertinents avec métadonnées
            
        Raises:
            ValueError: Si la requête est vide
            RuntimeError: Si erreur lors de la recherche
        """
        try:
            # Validation de la requête
            # Conversion sécurisée en string
            if query is None:
                raise ValueError("La requête ne peut pas être None")
                
            if not isinstance(query, str):
                query = str(query)
                
            if not query or not query.strip():
                raise ValueError("La requête ne peut pas être vide")

            query = query.strip()
            self.logger.info(f"Recherche de chunks pour la requête : '{query[:100]}...'")
            
            # Génération de l'embedding de la requête
            query_embedding = self.embedding_processor.generate_query_embedding(query)
            if query_embedding is None:
                raise RuntimeError("Impossible de générer l'embedding de la requête")
            
            # Utiliser la méthode unifiée du VectorStoreManager
            if hasattr(self.vector_store, 'search_similar'):
                # Essayer d'abord avec paramètre nommé
                try:
                    search_results = self.vector_store.search_similar(query_embedding, k=self.top_k)
                except TypeError:
                    # Fallback: paramètre positionnel
                    search_results = self.vector_store.search_similar(query_embedding, self.top_k)
            else:
                search_results = self.vector_store.search(query_embedding, self.top_k)

            if not search_results:
                self.logger.warning("Aucun résultat trouvé pour la requête")
                return []
            
            # Formatage des résultats
            retrieval_results = []
            for embedded_chunk, similarity_score in search_results:
                # CORRECTION: Extraire le chunk original depuis l'EmbeddedChunk
                if hasattr(embedded_chunk, 'chunk') and embedded_chunk.chunk:
                    actual_chunk = embedded_chunk.chunk
                else:
                    # Fallback: créer un DocumentChunk complet avec tous les paramètres requis
                    from processing.chunking import DocumentChunk
                    metadata = getattr(embedded_chunk, 'metadata', {})
                    actual_chunk = DocumentChunk(
                        content=getattr(embedded_chunk, 'text', ''),
                        metadata=metadata,
                        chunk_id=getattr(embedded_chunk, 'chunk_id', 'unknown'),
                        source_file=metadata.get('filename', 'unknown'),
                        page_number=metadata.get('page_number', 1),
                        chunk_index=0
                    )

                # Création de la référence source
                source_ref = self.format_source_reference(actual_chunk)

                # Normalisation des métadonnées pour compatibilité generator.py
                normalized_chunk = self._normalize_metadata_for_generator(actual_chunk)
                
                # AJOUT MANQUANT : Création du résultat
                result = RetrievalResult(
                    chunk=normalized_chunk,
                    similarity_score=float(similarity_score),
                    source_reference=source_ref
                )
                
                retrieval_results.append(result) 
            
            # Tri par score de similarité décroissant
            retrieval_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            self.logger.info(f"Récupéré {len(retrieval_results)} chunks pertinents")
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche : {e}")
            raise RuntimeError(f"Erreur lors de la recherche de chunks : {e}")
    
    def _normalize_metadata_for_generator(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Normalise les métadonnées pour compatibilité avec generator.py
        
        NOUVEAU : Préserve le flag is_metadata pour le generator
        """
        # Copie des métadonnées existantes
        normalized_metadata = chunk.metadata.copy() if hasattr(chunk, 'metadata') else {}
        
        # NOUVEAU : Préservation du flag is_metadata
        if 'is_metadata' in chunk.metadata:
            normalized_metadata['is_metadata'] = chunk.metadata['is_metadata']
        
        # Ajout des clés attendues par generator.py
        if hasattr(chunk, 'source_file'):
            normalized_metadata['filename'] = chunk.source_file
            if chunk.source_file.lower().endswith('.pdf'):
                normalized_metadata['source'] = chunk.source_file[:-4]
            else:
                normalized_metadata['source'] = chunk.source_file
        
        if hasattr(chunk, 'page_number'):
            normalized_metadata['page_number'] = chunk.page_number
            normalized_metadata['page'] = chunk.page_number
        
        # Création d'un nouveau chunk avec métadonnées normalisées
        try:
            normalized_chunk = DocumentChunk(
                content=chunk.content,
                metadata=normalized_metadata,
                chunk_id=getattr(chunk, 'chunk_id', 'unknown'),
                source_file=getattr(chunk, 'source_file', 'unknown'),
                page_number=getattr(chunk, 'page_number', 1),
                chunk_index=getattr(chunk, 'chunk_index', 0)
            )
            return normalized_chunk
        except Exception as e:
            self.logger.error(f"Erreur lors de la normalisation : {e}")
            return chunk
        
    # Méthodes d'alias pour compatibilité avec différents modules
    def retrieve_documents(self, query: str) -> List[RetrievalResult]:
        """
        Alias pour retrieve() - compatibilité avec certains modules
        
        Args:
            query: Requête utilisateur
            
        Returns:
            List[RetrievalResult]: Résultats de retrieval
        """
        return self.retrieve(query)
    
    def search(self, query: str) -> List[RetrievalResult]:
        """
        Alias pour retrieve() - compatibilité avec certains modules
        
        Args:
            query: Requête utilisateur
            
        Returns:
            List[RetrievalResult]: Résultats de retrieval
        """
        return self.retrieve(query)
    
    def get_relevant_documents(self, query: str) -> List[RetrievalResult]:
        """
        Alias pour retrieve() - compatibilité avec certains modules
        
        Args:
            query: Requête utilisateur
            
        Returns:
            List[RetrievalResult]: Résultats de retrieval
        """
        return self.retrieve(query)
    
    def format_source_reference(self, chunk: DocumentChunk) -> str:
        """
        Formate la référence source au format "fichier_page" ou "fichier_meta"
        
        CORRIGÉ : Assure la cohérence entre retrieval et generation
        
        Args:
            chunk: Chunk dont on veut la référence
            
        Returns:
            str: Référence formatée "fichier_page" ou "fichier_meta"
        """
        try:
            # 1. EXTRACTION NORMALISÉE DU FICHIER
            filename = None
            
            # Priorité aux champs standardisés
            if 'source' in chunk.metadata and chunk.metadata['source']:
                filename = chunk.metadata['source']
            elif 'filename' in chunk.metadata and chunk.metadata['filename']:
                filename = chunk.metadata['filename']
            elif hasattr(chunk, 'source_file') and chunk.source_file:
                filename = chunk.source_file
            else:
                filename = "fichier_inconnu"
            
            # Nettoyage systématique de l'extension .pdf
            if filename.lower().endswith('.pdf'):
                filename = filename[:-4]
            
            # 2. DÉTECTION DU TYPE DE CHUNK (CRUCIAL)
            is_metadata_chunk = chunk.metadata.get('is_metadata', False)
            
            if is_metadata_chunk:
                # MÉTADONNÉES : toujours "_meta"
                source_ref = f"{filename}_meta"
                self.logger.debug(f"Chunk métadonnées détecté : {source_ref}")
            else:
                # CONTENU : récupération cohérente du numéro de page
                page_num = self._extract_page_number_unified(chunk)
                source_ref = f"{filename}_{page_num}"
                self.logger.debug(f"Chunk contenu détecté : {source_ref} (page {page_num})")
            
            return source_ref
            
        except Exception as e:
            self.logger.error(f"Erreur formatage référence pour chunk {getattr(chunk, 'chunk_id', 'unknown')}: {e}")
            return "document_inconnu_1"

    def _extract_page_number_unified(self, chunk: DocumentChunk) -> int:
        """
        NOUVELLE MÉTHODE : Extraction unifiée et cohérente du numéro de page
        
        Priorité aux sources les plus fiables pour éviter les incohérences
        """
        # 1. Source principale : page_number (le plus fiable)
        if hasattr(chunk, 'page_number') and chunk.page_number:
            try:
                page_num = int(chunk.page_number)
                if page_num > 0:
                    return page_num
            except (ValueError, TypeError):
                pass
        
        # 2. Métadonnées : ordre de priorité établi
        metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
        
        # Ordre de priorité pour éviter les conflits
        priority_fields = ['page_number', 'page', 'page_num', 'pageNumber']
        
        for field in priority_fields:
            if field in metadata and metadata[field] is not None:
                try:
                    page_num = int(metadata[field])
                    if page_num > 0:
                        self.logger.debug(f"Page extraite du champ '{field}': {page_num}")
                        return page_num
                except (ValueError, TypeError):
                    continue
        
        # 3. Fallback : page 1 par défaut
        self.logger.debug("Aucun numéro de page trouvé, utilisation de la page 1 par défaut")
        return 1
    
    def format_sources_list(self, retrieval_results: List[RetrievalResult]) -> str:
        """
        Formate la liste des sources pour l'export CSV
        
        Crée une chaîne formatée contenant toutes les références sources
        selon le format requis pour le fichier CSV de résultats.
        
        Args:
            retrieval_results: Résultats de retrieval
            
        Returns:
            str: Sources formatées "[fichier_page, fichier_page, ...]"
            
        Examples:
            "[20240627_ELCA_Vincent_Schaller_entretien_FR_1, 20240611_ELCA_Hakan_Birsel_entretien_FR_8]"
        """
        if not retrieval_results:
            return "[]"
        
        try:
            # Extraction des références sources
            source_refs = [result.source_reference for result in retrieval_results]
            
            # Suppression des doublons tout en préservant l'ordre
            unique_sources = []
            seen = set()
            for source in source_refs:
                if source not in seen:
                    unique_sources.append(source)
                    seen.add(source)
            
            # Formatage de la liste
            sources_str = ", ".join(unique_sources)
            formatted_sources = f"[{sources_str}]"
            
            return formatted_sources
            
        except Exception as e:
            self.logger.error(f"Erreur lors du formatage des sources : {e}")
            return "[]"
    
    def filter_by_metadata(self, retrieval_results: List[RetrievalResult], 
                          metadata_filters: Dict[str, Any]) -> List[RetrievalResult]:
        """
        Filtre les résultats selon des critères de métadonnées
        
        Permet de filtrer les chunks récupérés selon des critères
        comme la langue, la date d'entretien, le nom de l'interviewé, etc.
        
        Args:
            retrieval_results: Résultats à filtrer
            metadata_filters: Critères de filtrage (clé: valeur)
            
        Returns:
            List[RetrievalResult]: Résultats filtrés
            
        Examples:
            >>> filters = {'custominterviewlanguage': 'français'}
            >>> filtered_results = retriever.filter_by_metadata(results, filters)
        """
        if not metadata_filters or not retrieval_results:
            return retrieval_results
        
        try:
            filtered_results = []
            
            for result in retrieval_results:
                matches_all_filters = True
                
                for filter_key, filter_value in metadata_filters.items():
                    chunk_metadata = result.chunk.metadata
                    
                    # Vérification de l'existence de la clé
                    if filter_key not in chunk_metadata:
                        matches_all_filters = False
                        break
                    
                    chunk_value = chunk_metadata[filter_key]
                    
                    # Comparaison flexible (insensible à la casse pour les chaînes)
                    if isinstance(filter_value, str) and isinstance(chunk_value, str):
                        if filter_value.lower() not in chunk_value.lower():
                            matches_all_filters = False
                            break
                    else:
                        if chunk_value != filter_value:
                            matches_all_filters = False
                            break
                
                if matches_all_filters:
                    filtered_results.append(result)
            
            self.logger.info(f"Filtré {len(filtered_results)}/{len(retrieval_results)} résultats")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors du filtrage : {e}")
            return retrieval_results  # Retourne les résultats non filtrés en cas d'erreur
    
    def get_context_for_generation(self, retrieval_results: List[RetrievalResult]) -> str:
        """
        Combine les chunks récupérés en contexte pour la génération
        
        NOUVEAU : Marque distinctement les chunks de métadonnées et de contenu
        """
        if not retrieval_results:
            return ""
        
        try:
            context_parts = []
            
            for i, result in enumerate(retrieval_results, 1):
                chunk = result.chunk
                source_ref = result.source_reference
                similarity = result.similarity_score
                
                # NOUVEAU : Identification du type de chunk
                is_metadata_chunk = chunk.metadata.get('is_metadata', False)
                chunk_type = "MÉTADONNÉES" if is_metadata_chunk else "CONTENU"
                
                # Formatage du chunk avec identification du type
                chunk_context = f"--- Document {i} ({chunk_type}) (Score: {similarity:.3f}) ---\n"
                chunk_context += f"Source: {source_ref}\n"
                
                # Ajout des métadonnées importantes seulement pour les chunks de contenu
                if not is_metadata_chunk:
                    metadata = chunk.metadata
                    if 'interviewee_name' in metadata:
                        chunk_context += f"Interviewé: {metadata['interviewee_name']}\n"
                    if 'interview_date' in metadata:
                        chunk_context += f"Date: {metadata['interview_date']}\n"
                    if 'interview_language' in metadata:
                        chunk_context += f"Langue: {metadata['interview_language']}\n"
                
                chunk_context += f"Contenu:\n{chunk.content}\n\n"
                context_parts.append(chunk_context)
            
            # Assemblage du contexte complet
            full_context = "Contexte documentaire pour répondre à la question :\n\n"
            full_context += "".join(context_parts)
            
            # Limitation de la taille du contexte si nécessaire
            max_context_length = 8000
            if len(full_context) > max_context_length:
                self.logger.warning("Contexte tronqué pour respecter les limites")
                full_context = full_context[:max_context_length] + "\n... [contexte tronqué]"
            
            return full_context
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du contexte : {e}")
            return "Erreur lors de la création du contexte documentaire."
    
    def get_retrieval_statistics(self, retrieval_results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Génère des statistiques sur les résultats de retrieval
        
        Args:
            retrieval_results: Résultats de retrieval
            
        Returns:
            Dict[str, Any]: Statistiques des résultats
        """
        if not retrieval_results:
            return {
                'total_results': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'unique_sources': 0
            }
        
        scores = [result.similarity_score for result in retrieval_results]
        sources = set(result.source_reference for result in retrieval_results)
        
        return {
            'total_results': len(retrieval_results),
            'avg_similarity': np.mean(scores),
            'max_similarity': max(scores),
            'min_similarity': min(scores),
            'unique_sources': len(sources),
            'source_distribution': dict([(src, sum(1 for r in retrieval_results if r.source_reference == src)) for src in sources])
        }
