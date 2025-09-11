from typing import List, Dict, Any, Optional
import numpy as np
import logging
import requests
import json
import time
from dataclasses import dataclass
from tqdm import tqdm

from processing.chunking import DocumentChunk


@dataclass
class EmbeddedChunk:
    """
    Représente un chunk de document avec son vecteur d'embedding
    
    Attributes:
        chunk: Le chunk de document original
        embedding_vector: Vecteur d'embedding généré par le modèle
        embedding_model: Nom du modèle utilisé pour l'embedding
    """
    chunk: DocumentChunk
    embedding_vector: np.ndarray
    embedding_model: str


class EmbeddingProcessor:
    """
    Processeur d'embeddings utilisant Ollama pour vectoriser les chunks de documents
    
    Gère la génération d'embeddings pour les chunks et les requêtes utilisateur
    en utilisant les modèles d'embedding disponibles via Ollama.
    """
    
    def __init__(self, embedding_model: str, batch_size: int = 50, delay_between_batches: float = 0.1):
        """
        Initialise le processeur d'embeddings
        
        Args:
            embedding_model: Nom du modèle d'embedding à utiliser
            batch_size: Nombre de chunks à traiter par batch
            delay_between_batches: Délai en secondes entre les batches
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.ollama_client = None
        self.logger = logging.getLogger(__name__)
        self.ollama_base_url = "http://localhost:11434"
        self._embedding_dimension = None
        self.max_retries = 3
        self.request_timeout = 30
        
        # Initialisation du client Ollama
        self.initialize_ollama_client()
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Génère l'embedding pour une requête de recherche
    
        Args:
            query: Requête utilisateur à transformer en embedding
        
        Returns:
            np.ndarray: Vecteur d'embedding de la requête
        """
        try:
            if not query or not query.strip():
                raise ValueError("La requête ne peut pas être vide")
        
            # Validation du client Ollama
            if not self.ollama_client and not self.initialize_ollama_client():
                raise RuntimeError("Impossible d'initialiser le client Ollama pour l'embedding")
        
            self.logger.debug(f"Génération d'embedding pour la requête : {query[:50]}...")
        
            embedding_vector = self._generate_embedding_with_retry(query.strip())
            
            # DIAGNOSTIC - Test embedding généré (APRÈS la génération)
            print(f"DIAGNOSTIC - Query embedding generated: {embedding_vector.shape[0] if hasattr(embedding_vector, 'shape') else 'NO SHAPE'} dimensions")
        
            return embedding_vector
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de l'embedding de requête : {e}")
            raise RuntimeError(f"Impossible de générer l'embedding pour la requête : {e}")

    def initialize_ollama_client(self) -> bool:
        """
        Initialise le client Ollama pour l'embedding
    
        Returns:
            bool: True si l'initialisation réussit
        """
        try:
            self.logger.info(f"Initialisation du client Ollama pour le modèle {self.embedding_model_name}")
            
            # Test de connexion via API REST
            url = f"{self.ollama_base_url}/api/tags"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"Ollama non accessible : status {response.status_code}")
                    return False
                    
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                # Vérification que le modèle d'embedding est disponible
                model_found = False
                for model_name in available_models:
                    if self.embedding_model_name in model_name:
                        model_found = True
                        self.logger.info(f"Modèle d'embedding trouvé : {model_name}")
                        break
                
                if not model_found:
                    self.logger.error(f"Modèle d'embedding {self.embedding_model_name} non trouvé dans : {available_models}")
                    return False
                
                self.ollama_client = True
                self.logger.info(f"Client Ollama initialisé avec succès")
                return True
                
            except requests.RequestException as e:
                self.logger.error(f"Erreur de connexion à Ollama : {e}")
                return False
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation d'Ollama : {e}")
            return False
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """
        Génère les embeddings pour tous les chunks avec traitement par batch
        """
        try:
            self.logger.info(f"Génération des embeddings pour {len(chunks)} chunks (batch_size: {self.batch_size})")
            embedded_chunks = []
            failed_chunks = 0
            
            # Filtrage préalable des chunks valides
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                if not chunk or not chunk.content or not chunk.content.strip():
                    self.logger.warning(f"Chunk {i+1} vide ou invalide, ignoré")
                    continue
                    
                if len(chunk.content.strip()) < 10:
                    self.logger.warning(f"Chunk {i+1} trop court ({len(chunk.content)} chars), ignoré")
                    continue
                    
                valid_chunks.append((i, chunk))
            
            self.logger.info(f"{len(valid_chunks)} chunks valides à traiter sur {len(chunks)} total")
            
            # Traitement par batch avec barre de progression
            with tqdm(total=len(valid_chunks), desc="Génération embeddings") as pbar:
                for batch_start in range(0, len(valid_chunks), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(valid_chunks))
                    batch = valid_chunks[batch_start:batch_end]
                    
                    self.logger.debug(f"Traitement batch {batch_start//self.batch_size + 1}/{(len(valid_chunks) + self.batch_size - 1)//self.batch_size}")
                    
                    # Traitement du batch
                    batch_results = self._process_batch(batch)
                    embedded_chunks.extend(batch_results)
                    
                    # Mise à jour de la barre de progression
                    pbar.update(len(batch))
                    
                    # Délai entre les batches pour éviter la surcharge
                    if batch_end < len(valid_chunks):
                        time.sleep(self.delay_between_batches)
        
            successful_chunks = len(embedded_chunks)
            self.logger.info(f"Embeddings générés avec succès : {successful_chunks}/{len(valid_chunks)} chunks valides")
            
            if failed_chunks > 0:
                self.logger.warning(f"{failed_chunks} chunks ont échoué lors du traitement")
            
            # DIAGNOSTIC - Vérification embeddings générés
            print(f"DIAGNOSTIC - Embeddings generated: {len(embedded_chunks)}/{len(valid_chunks)} chunks")
            if embedded_chunks:
                print(f"DIAGNOSTIC - First embedding dimension: {embedded_chunks[0].embedding_vector.shape[0] if hasattr(embedded_chunks[0].embedding_vector, 'shape') else 'NO SHAPE'}")

            return embedded_chunks
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des embeddings : {e}")
            raise
    
    def _process_batch(self, batch: List[tuple]) -> List[EmbeddedChunk]:
        """
        Traite un batch de chunks
        
        Args:
            batch: Liste de tuples (index, chunk) à traiter
            
        Returns:
            List[EmbeddedChunk]: Chunks avec embeddings générés
        """
        batch_results = []
        
        for chunk_index, chunk in batch:
            try:
                embedded_chunk = self._embed_single_chunk_safe(chunk)
                if embedded_chunk:
                    batch_results.append(embedded_chunk)
                else:
                    self.logger.warning(f"Embedding invalide pour le chunk {chunk_index + 1}")
                    
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement du chunk {chunk_index + 1} : {e}")
                continue
        
        return batch_results
    
    def _embed_single_chunk_safe(self, chunk: DocumentChunk) -> Optional[EmbeddedChunk]:
        """
        Version sécurisée de embed_single_chunk avec gestion d'erreurs
        """
        try:
            return self.embed_single_chunk(chunk)
        except Exception as e:
            self.logger.error(f"Erreur lors de l'embedding du chunk : {e}")
            return None
    
    def embed_single_chunk(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """
        Génère l'embedding pour un chunk individuel
        
        Args:
            chunk: Chunk à vectoriser
            
        Returns:
            EmbeddedChunk: Chunk avec son vecteur d'embedding
        """
        try:
            # Préparation du texte à vectoriser
            text_to_embed = chunk.content.strip()
            
            # Appel à l'API Ollama pour générer l'embedding
            embedding_vector = self._generate_embedding_with_retry(text_to_embed)
            
            # Création de l'EmbeddedChunk
            embedded_chunk = EmbeddedChunk(
                chunk=chunk,
                embedding_vector=embedding_vector,
                embedding_model=self.embedding_model_name
            )
            
            return embedded_chunk
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de l'embedding pour le chunk : {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Génère l'embedding pour une requête utilisateur
        
        Args:
            query: Requête à vectoriser
            
        Returns:
            np.ndarray: Vecteur d'embedding de la requête
        """
        try:
            self.logger.debug(f"Génération de l'embedding pour la requête : {query[:50]}...")
            
            if not query or not query.strip():
                raise ValueError("La requête ne peut pas être vide")
            
            embedding_vector = self._generate_embedding_with_retry(query.strip())
            
            self.logger.debug(f"Embedding de requête généré avec dimension {embedding_vector.shape[0]}")
            return embedding_vector
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de l'embedding pour la requête : {e}")
            raise
    
    def _generate_embedding_with_retry(self, text: str) -> np.ndarray:
        """
        Génère un embedding avec retry automatique en cas d'échec
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return self._generate_embedding(text)
                
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 0.5  # Attente progressive
                    self.logger.warning(f"Tentative {attempt + 1}/{self.max_retries} échouée, retry dans {wait_time}s : {e}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Toutes les tentatives ont échoué pour générer l'embedding")
        
        raise last_exception
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Génère un embedding via l'API Ollama
        """
        try:
            url = f"{self.ollama_base_url}/api/embeddings"
            payload = {
                "model": self.embedding_model_name,
                "prompt": text
            }
        
            self.logger.debug(f"Requête embedding vers {url} avec modèle {self.embedding_model_name}")
        
            # Envoi de la requête avec timeout
            response = requests.post(
                url,
                json=payload,
                timeout=self.request_timeout
            )
            
            # Vérification du statut de la réponse
            if response.status_code != 200:
                raise Exception(f"Erreur API Ollama : {response.status_code} - {response.text}")
        
            # Vérification du contenu de la réponse
            if not response.text or not response.text.strip():
                raise Exception("Réponse Ollama vide")
        
            # Parsing JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                self.logger.error(f"Réponse JSON invalide : {response.text[:500]}")
                raise Exception(f"Réponse JSON invalide d'Ollama : {e}")
        
            # Extraction de l'embedding
            if 'embedding' in response_data:
                embedding = response_data['embedding']
                if not embedding:
                    raise Exception("Embedding vide retourné par Ollama")
            else:
                if 'embeddings' in response_data and response_data['embeddings']:
                    embedding = response_data['embeddings'][0]
                else:
                    available_keys = list(response_data.keys()) if isinstance(response_data, dict) else 'N/A'
                    raise Exception(f"Clé 'embedding' non trouvée. Clés disponibles : {available_keys}")
        
            # Conversion en numpy array
            embedding_vector = np.array(embedding, dtype=np.float32)
            
            if embedding_vector.size == 0:
                raise Exception("Vecteur d'embedding vide")
                
            return embedding_vector
        
        except requests.Timeout:
            raise Exception(f"Timeout lors de la requête embedding (>{self.request_timeout}s)")
        except requests.ConnectionError:
            raise Exception("Erreur de connexion à Ollama")
        except Exception as e:
            raise Exception(f"Erreur lors de la génération d'embedding : {e}")
    
    def validate_embedding_model(self) -> bool:
        """
        Vérifie que le modèle d'embedding est disponible dans Ollama
    
        Returns:
            bool: True si le modèle est disponible, False sinon
        """
        try:
            self.logger.debug(f"Validation du modèle d'embedding {self.embedding_model_name}")
        
            # Vérification via l'API des modèles disponibles
            url = f"{self.ollama_base_url}/api/tags"
            response = requests.get(url, timeout=10)
        
            if response.status_code != 200:
                self.logger.error(f"Impossible d'accéder à l'API Ollama : {response.status_code}")
                return False
        
            # Analyse de la réponse
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
        
            # Vérification de la présence du modèle
            model_available = any(
                self.embedding_model_name in model_name 
                for model_name in available_models
            )
        
            if model_available:
                self.logger.info(f"Modèle d'embedding {self.embedding_model_name} validé")
            
                # Test avec un embedding simple
                try:
                    test_embedding = self._generate_embedding("test")
                    if isinstance(test_embedding, np.ndarray) and test_embedding.size > 0:
                        self._embedding_dimension = len(test_embedding)
                        self.logger.info(f"Test d'embedding réussi, dimension : {self._embedding_dimension}")
                        return True
                    else:
                        self.logger.error(f"Test d'embedding échoué : réponse = {test_embedding}")
                        return False
                except Exception as e:
                    self.logger.error(f"Impossible de tester l'embedding : {e}")
                    return False
            else:
                self.logger.error(f"Modèle d'embedding {self.embedding_model_name} non disponible")
                return False
                
        except requests.RequestException as e:
            self.logger.error(f"Erreur de communication lors de la validation : {e}")
            return False
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation du modèle : {e}")
            return False
    
    def get_embedding_dimension(self) -> int:
        """
        Récupère la dimension des vecteurs d'embedding du modèle
        
        Returns:
            int: Dimension des vecteurs d'embedding
        """
        try:
            if self._embedding_dimension is not None:
                return self._embedding_dimension
            
            self.logger.debug("Détermination de la dimension des embeddings")
            
            test_text = "test dimension embedding"
            test_embedding = self._generate_embedding(test_text)
            
            self._embedding_dimension = len(test_embedding)
            self.logger.info(f"Dimension des embeddings déterminée : {self._embedding_dimension}")
            
            return self._embedding_dimension
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détermination de la dimension : {e}")
            raise Exception(f"Impossible de déterminer la dimension des embeddings : {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle d'embedding actuel
        """
        return {
            'model_name': self.embedding_model_name,
            'embedding_dimension': self._embedding_dimension,
            'ollama_url': self.ollama_base_url,
            'client_initialized': self.ollama_client is not None,
            'model_validated': self._embedding_dimension is not None,
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'request_timeout': self.request_timeout
        }
    
    def cleanup(self) -> None:
        """
        Nettoie les ressources utilisées par le processeur
        """
        self.logger.info("Nettoyage du processeur d'embeddings")
        self.ollama_client = None
        self._embedding_dimension = None
    
    # Méthode de compatibilité
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """
        Alias pour embed_chunks pour compatibilité
        """
        return self.embed_chunks(chunks)
