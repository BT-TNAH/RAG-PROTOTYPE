from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import hashlib
import re
from pathlib import Path

# Import du module de chargement PDF
from data_loading.pdf_loader import LoadedDocument
from data_loading.pdf_loader import LoadedDocument, PDFLoader

@dataclass
class DocumentChunk:
    """
    Représente un chunk (fragment) d'un document avec ses métadonnées
    
    Attributes:
        content (str): Contenu textuel du chunk
        metadata (Dict[str, Any]): Métadonnées héritées du document source + métadonnées spécifiques au chunk
        chunk_id (str): Identifiant unique du chunk
        source_file (str): Nom du fichier source (ex: "20240627_ELCA_Vincent_Schaller_entretien_FR")
        page_number (int): Numéro de page estimé du chunk dans le document original
        chunk_index (int): Index du chunk dans la séquence de chunks du document (commence à 0)
    """
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int


class ChunkingProcessor:
    """
    Processeur de chunking pour découper les documents en fragments selon la configuration
    
    Supporte le chunking à taille fixe avec overlap configurable.
    Préserve les métadonnées des documents sources et estime les numéros de pages.
    """
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initialise le processeur de chunking
        
        Args:
            chunk_size (int): Taille maximale d'un chunk en caractères
            chunk_overlap (int): Nombre de caractères de chevauchement entre chunks consécutifs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        # NOUVEAU : Instance de PDFLoader pour accès aux métadonnées
        self.pdf_loader = PDFLoader()
        
        # Validation des paramètres
        if chunk_size <= 0:
            raise ValueError("chunk_size doit être positif")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap ne peut pas être négatif")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap doit être inférieur à chunk_size")
        
        self.logger.info(f"ChunkingProcessor initialisé - taille: {chunk_size}, overlap: {chunk_overlap}")
    
    def chunk_documents(self, documents: List[LoadedDocument]) -> List[DocumentChunk]:
        """
        Découpe tous les documents en chunks selon la configuration
        
        Args:
            documents (List[LoadedDocument]): Liste des documents à découper
            
        Returns:
            List[DocumentChunk]: Liste de tous les chunks créés, triés par document puis par index
            
        Raises:
            ValueError: Si la liste de documents est vide
            Exception: Si une erreur survient lors du chunking d'un document
        """
        if not documents:
            raise ValueError("La liste de documents ne peut pas être vide")
        
        self.logger.info(f"Début du chunking de {len(documents)} documents")
        all_chunks = []
        
        for doc_index, document in enumerate(documents):
            try:
                # CORRECTION: Utiliser document.metadata.filename
                self.logger.debug(f"Chunking du document {doc_index + 1}/{len(documents)}: {document.metadata.filename}")
                doc_chunks = self.chunk_single_document(document)
                all_chunks.extend(doc_chunks)
                # DIAGNOSTIC - Vérification chunking par document
                print(f"DIAGNOSTIC - Document {document.metadata.filename}: {len(doc_chunks)} chunks created")
                self.logger.debug(f"Document {document.metadata.filename}: {len(doc_chunks)} chunks créés")
                
            except Exception as e:
                # CORRECTION: Utiliser document.metadata.filename
                self.logger.error(f"Erreur lors du chunking du document {document.metadata.filename}: {e}")
                raise Exception(f"Échec du chunking du document {document.metadata.filename}: {e}")
        
        self.logger.info(f"Chunking terminé - {len(all_chunks)} chunks créés au total")
        # DIAGNOSTIC - Vérification chunking total
        print(f"DIAGNOSTIC - Total chunks created: {len(all_chunks)}")
        if all_chunks:
            print(f"DIAGNOSTIC - First chunk length: {len(all_chunks[0].content)}")
            print(f"DIAGNOSTIC - Last chunk length: {len(all_chunks[-1].content)}")
        return all_chunks
    
    def chunk_single_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        """
        Découpe un document individuel en chunks de contenu ET crée un chunk de métadonnées
        
        Args:
            document (LoadedDocument): Document à découper
        
        Returns:
            List[DocumentChunk]: Liste des chunks (contenu + métadonnées), ordonnés par chunk_index
        
        Raises:
            ValueError: Si le document est invalide ou vide
        """
        if not document or not document.content.strip():
            filename = document.metadata.filename if document and document.metadata else 'None'
            raise ValueError(f"Document invalide ou vide: {filename}")

        self.logger.debug(f"Chunking du document {document.metadata.filename} ({len(document.content)} caractères)")

        # Extraction des métadonnées de base
        base_metadata = {
            'filename': document.metadata.filename,
            'file_path': document.metadata.file_path,
            'total_pages': document.metadata.total_pages,
            'content_length': len(document.content),
            'interviewee_name': document.metadata.custom_interviewee_name,
            'interview_date': document.metadata.custom_interview_date,
            'interview_language': document.metadata.custom_interview_language,
            'archive_type': document.metadata.custom_archive_type
        }

        # NOUVEAU : Création du chunk de métadonnées AVANT les chunks de contenu
        metadata_chunks = self._create_metadata_chunk(document, base_metadata)
        
        # Validation du résultat de _create_metadata_chunk
        if metadata_chunks is None:
            self.logger.warning(f"_create_metadata_chunk a retourné None pour {document.metadata.filename}")
            metadata_chunks = []

        # Création des chunks de contenu (logique existante)
        content_chunks = self.create_fixed_size_chunks(document.content, base_metadata)

        # Combinaison des chunks : métadonnées d'abord, puis contenu
        all_chunks = metadata_chunks + content_chunks

        # Attribution des informations finales à tous les chunks
        source_file_stem = Path(document.metadata.filename).stem
        for i, chunk in enumerate(all_chunks):
            chunk.source_file = source_file_stem
            chunk.chunk_index = i
            if not hasattr(chunk.metadata, 'is_metadata') or not chunk.metadata.get('is_metadata', False):
                # Chunk de contenu : calcul de la page normale
                chunk.page_number = self.extract_page_information(chunk.content, document)
            else:
                # Chunk de métadonnées : page = "meta" (sera géré par les modules suivants)
                chunk.page_number = 1  # Valeur par défaut pour éviter les erreurs
            chunk.chunk_id = self.generate_chunk_id(chunk.source_file, chunk.chunk_index)

        self.logger.debug(f"Document {document.metadata.filename}: {len(all_chunks)} chunks générés ({len(metadata_chunks)} métadonnées + {len(content_chunks)} contenu)")
        
        return all_chunks
    
    def _create_metadata_chunk(self, document: LoadedDocument, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Crée un chunk contenant les métadonnées recherchables du document
        
        Args:
            document (LoadedDocument): Document source
            base_metadata (Dict[str, Any]): Métadonnées de base à enrichir
            
        Returns:
            List[DocumentChunk]: Liste contenant UN chunk de métadonnées
        """
        try:
            # Récupération du contenu de métadonnées recherchables
            metadata_content = self.pdf_loader.get_searchable_metadata_content(Path(document.metadata.file_path))
            
            if not metadata_content or not metadata_content.strip():
                self.logger.warning(f"Aucun contenu de métadonnées extrait pour {document.metadata.filename}")
                return []
            
            # Métadonnées spécifiques au chunk de métadonnées
            metadata_chunk_metadata = base_metadata.copy()
            metadata_chunk_metadata.update({
                'is_metadata': True,  # FLAG CRITIQUE
                'chunk_type': 'metadata',
                'source_type': 'document_metadata',
                'chunk_start_position': 0,
                'chunk_end_position': len(metadata_content),
                'chunk_size_actual': len(metadata_content),
                'chunk_overlap_used': 0
            })
            
            # Création du chunk de métadonnées
            metadata_chunk = DocumentChunk(
                content=metadata_content,
                metadata=metadata_chunk_metadata,
                chunk_id="",  # À définir plus tard
                source_file="",  # À définir plus tard  
                page_number=1,  # Valeur temporaire
                chunk_index=0   # Sera réajusté
            )
            
            return [metadata_chunk]
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du chunk de métadonnées pour {document.metadata.filename}: {e}")
            return []
    
    def create_fixed_size_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Crée des chunks de taille fixe avec métadonnées de position enrichies
        """
        if not text.strip():
            raise ValueError("Le texte à chunker ne peut pas être vide")

        chunks = []
        text_length = len(text)
        current_position = 0

        self.logger.debug(f"Création de chunks - texte: {text_length} caractères, "
                        f"taille: {self.chunk_size}, overlap: {self.chunk_overlap}")

        while current_position < text_length:
            # Détermination de la fin du chunk
            chunk_end = min(current_position + self.chunk_size, text_length)
            chunk_content = text[current_position:chunk_end]
        
            # Optimisation des coupures de mots
            if chunk_end < text_length and self.chunk_size > 100:
                last_space_offset = chunk_content.rfind(' ', max(0, len(chunk_content) - 50))
                if last_space_offset > len(chunk_content) * 0.8:
                    chunk_content = chunk_content[:last_space_offset]
                    chunk_end = current_position + last_space_offset
        
            # Nettoyage du chunk
            chunk_content = self._clean_chunk_content(chunk_content)
        
            # Métadonnées enrichies avec marquage de contenu
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'is_metadata': False,  # NOUVEAU : Marquage explicite comme chunk de contenu
                'chunk_type': 'content',  # NOUVEAU : Type explicite
                'chunk_start_position': current_position,
                'chunk_end_position': chunk_end,
                'chunk_size_actual': len(chunk_content),
                'chunk_overlap_used': min(self.chunk_overlap, current_position) if current_position > 0 else 0,
                'text_total_length': text_length,
                'chunk_position_ratio': current_position / text_length
            })
        
            # Création du chunk avec position stockée
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id="",  # À définir plus tard
                source_file="",  # À définir plus tard
                page_number=0,  # À définir plus tard
                chunk_index=len(chunks)  # Index temporaire
            )
        
            # Stockage de la position pour extract_page_information
            chunk._start_position = current_position
        
            chunks.append(chunk)
        
            # Calcul de la position suivante avec overlap
            if chunk_end >= text_length:
                break
        
            current_position = max(chunk_end - self.chunk_overlap, current_position + 1)

        self.logger.debug(f"Chunks de contenu créés: {len(chunks)}")
        return chunks
    
    def _clean_chunk_content(self, content: str) -> str:
        """
        Nettoie le contenu d'un chunk
        
        Args:
            content (str): Contenu brut du chunk
            
        Returns:
            str: Contenu nettoyé
        """
        if not content:
            return content
        
        # Suppression des espaces en début et fin
        content = content.strip()
        
        # Normalisation des espaces multiples
        content = re.sub(r'\s+', ' ', content)
        
        # Suppression des retours à la ligne multiples
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content
    
    def generate_chunk_id(self, source_file: str, chunk_index: int) -> str:
        """
        Génère un identifiant unique pour un chunk
        
        Args:
            source_file (str): Nom du fichier source sans extension
            chunk_index (int): Index du chunk dans le document
            
        Returns:
            str: Identifiant unique du chunk au format "source_file_chunk_index_hash"
        """
        if not source_file:
            raise ValueError("source_file ne peut pas être vide")
        if chunk_index < 0:
            raise ValueError("chunk_index doit être positif ou nul")
        
        # Création d'un hash basé sur le nom du fichier et l'index pour assurer l'unicité
        hash_input = f"{source_file}_{chunk_index}_{self.chunk_size}_{self.chunk_overlap}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        chunk_id = f"{source_file}_chunk_{chunk_index}_{hash_suffix}"
        return chunk_id
    
    def extract_page_information(self, chunk_content: str, document: LoadedDocument) -> int:
        """
        Détermine le numéro de page d'un chunk dans le document original
        """
        if not chunk_content or not document.content:
            return 1

        try:
            # CORRECTION: Méthode de recherche améliorée
            chunk_start_pos = -1
        
            # Méthode 1: Recherche avec échantillons de différentes tailles
            for sample_size in [50, 100, 200]:
                if len(chunk_content) >= sample_size:
                    sample = chunk_content[:sample_size].strip()
                    # Nettoyage des caractères problématiques
                    sample = re.sub(r'\s+', ' ', sample)
                    pos = document.content.find(sample)
                    if pos != -1:
                        chunk_start_pos = pos
                        break
        
            # Méthode 2: Si échec, recherche par mots-clés uniques
            if chunk_start_pos == -1:
                # Extraction de mots uniques du chunk (longueur > 4 caractères)
                words = re.findall(r'\b\w{5,}\b', chunk_content[:200])
                for word in words[:5]:  # Tester les 5 premiers mots
                    pos = document.content.find(word)
                    if pos != -1:
                        chunk_start_pos = pos
                        break
        
            # Méthode 3: Utilisation des métadonnées de position si disponibles
            if chunk_start_pos == -1 and 'chunk_start_position' in self.__dict__:
                chunk_start_pos = getattr(self, 'chunk_start_position', 0)
        
            if chunk_start_pos == -1:
                # Estimation basée sur l'index du chunk si disponible
                self.logger.debug(f"Utilisation de l'estimation par index pour {document.metadata.filename}")
                return 1
        
            # Calcul du ratio de position dans le document
            doc_length = len(document.content)
            position_ratio = chunk_start_pos / doc_length if doc_length > 0 else 0
        
            # Utilisation des métadonnées pour le calcul de page
            if hasattr(document.metadata, 'total_pages') and document.metadata.total_pages:
                estimated_page = max(1, int(position_ratio * document.metadata.total_pages) + 1)
                return min(estimated_page, document.metadata.total_pages)
            else:
                # Estimation: ~2000 caractères par page
                estimated_page = max(1, int(chunk_start_pos / 2000) + 1)
                return estimated_page
            
        except Exception as e:
            self.logger.warning(f"Erreur lors de l'estimation de la page pour le chunk: {e}")
            return 1
    
    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les chunks créés
        
        Args:
            chunks (List[DocumentChunk]): Liste des chunks à analyser
            
        Returns:
            Dict[str, Any]: Statistiques détaillées
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'documents_processed': 0
            }
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        unique_sources = set(chunk.source_file for chunk in chunks)
        
        stats = {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'avg_chunk_size': sum(chunk_sizes) / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'documents_processed': len(unique_sources),
            'chunks_per_document': len(chunks) / len(unique_sources) if unique_sources else 0,
            'chunk_size_config': self.chunk_size,
            'chunk_overlap_config': self.chunk_overlap
        }
        
        return stats
    
    def validate_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Valide la cohérence des chunks créés
        
        Args:
            chunks (List[DocumentChunk]): Liste des chunks à valider
            
        Returns:
            List[str]: Liste des erreurs de validation (vide si tout est OK)
        """
        errors = []
        
        if not chunks:
            errors.append("Aucun chunk à valider")
            return errors
        
        # Validation des IDs uniques
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            errors.append("Des chunk_ids dupliqués ont été détectés")
        
        # Validation des chunks individuels
        for i, chunk in enumerate(chunks):
            if not chunk.content.strip():
                errors.append(f"Chunk {i} a un contenu vide")
            
            if not chunk.chunk_id:
                errors.append(f"Chunk {i} n'a pas d'ID")
            
            if not chunk.source_file:
                errors.append(f"Chunk {i} n'a pas de source_file")
            
            if chunk.page_number < 1:
                errors.append(f"Chunk {i} a un numéro de page invalide: {chunk.page_number}")
            
            if chunk.chunk_index < 0:
                errors.append(f"Chunk {i} a un index invalide: {chunk.chunk_index}")
        
        return errors


# Alias pour compatibilité avec l'architecture existante
# (le main.py importe DocumentChunker)
class DocumentChunker(ChunkingProcessor):
    """
    Alias pour ChunkingProcessor pour compatibilité avec main.py
    """
    pass
