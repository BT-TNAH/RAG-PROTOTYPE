from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from pathlib import Path
import logging
import re
from datetime import datetime

# Importation de PyPDF2 pour la lecture des PDFs et métadonnées
try:
    import PyPDF2
except ImportError:
    raise ImportError(
        "PyPDF2 est requis pour le chargement des PDFs. "
        "Installez-le avec: pip install PyPDF2"
    )

@dataclass
class DocumentMetadata:
    """
    Métadonnées personnalisées d'un document ELCA
    
    CORRECTION: Ajout des attributs manquants utilisés dans chunking.py
    """
    custom_interviewee_name: str
    custom_interview_date: str
    custom_interview_language: str
    custom_archive_type: str
    filename: str
    file_path: str
    # CORRECTION: Ajout de total_pages utilisé dans chunking.py ligne 221
    total_pages: int = 1

@dataclass
class LoadedDocument:
    """
    Document chargé avec son contenu et métadonnées
    
    CORRECTION: Cohérence des attributs avec chunking.py
    """
    content: str
    metadata: DocumentMetadata
    # CORRECTION: Suppression de page_count au niveau LoadedDocument
    # car il existe déjà dans DocumentMetadata.total_pages
    # CORRECTION: Suppression de filename car il existe déjà dans metadata.filename

class PDFLoader:
    """
    Chargeur de documents PDF pour le prototype RAG ELCA
    
    Gère le chargement des 19 transcriptions d'entretiens avec extraction
    des métadonnées personnalisées et validation de la structure.
    """
    
    def __init__(self, data_directory: str = "data/"):
        """
        Initialise le chargeur PDF
        
        Args:
            data_directory: Chemin vers le dossier contenant les PDFs
        """
        self.data_directory = Path(data_directory)
        self.logger = logging.getLogger(__name__)
        
        # Patterns pour l'extraction des métadonnées depuis les noms de fichiers
        self.filename_pattern = re.compile(
            r'(\d{8})_ELCA_([^_]+)_([^_]+)_entretien_([A-Z]{2})(?:_(\d+))?'
        )
        
        # Mapping des codes langue
        self.language_mapping = {
            'FR': 'français',
            'AN': 'anglais'
        }
        
        # Extensions PDF supportées
        self.supported_extensions = {'.pdf', '.PDF'}

    def load_documents(self, data_directory: str | Path) -> List[LoadedDocument]:
        """
        Méthode wrapper pour maintenir la compatibilité avec main.py
        
        Args:
            data_directory: Chemin vers le dossier contenant les PDFs (peut être ignoré si déjà défini)
        
        Returns:
            List[LoadedDocument]: Liste de tous les documents chargés avec métadonnées
        """
        # Si un nouveau dossier est spécifié, on l'utilise
        if data_directory and str(data_directory) != str(self.data_directory):
            self.data_directory = Path(data_directory)
            self.logger.info(f"Dossier de données mis à jour : {self.data_directory}")
        
        return self.load_all_pdfs()
    
    def load_all_pdfs(self) -> List[LoadedDocument]:
        """
        Charge tous les fichiers PDF du dossier data/
        
        Returns:
            List[LoadedDocument]: Liste de tous les documents chargés avec métadonnées
            
        Raises:
            FileNotFoundError: Si le dossier data/ n'existe pas
            ValueError: Si aucun PDF valide n'est trouvé
        """
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Le dossier {self.data_directory} n'existe pas")
        
        self.logger.info(f"Recherche des fichiers PDF dans {self.data_directory}")
        
        # Récupération de la liste des fichiers PDF
        pdf_files = self.get_pdf_files_list()
        
        if not pdf_files:
            raise ValueError("Aucun fichier PDF trouvé dans le dossier data/")
        
        self.logger.info(f"Trouvé {len(pdf_files)} fichiers PDF à traiter")
        
        # Chargement séquentiel des fichiers
        loaded_documents = []
        errors = []
        
        for pdf_file in pdf_files:
            try:
                document = self.load_single_pdf(pdf_file)
                loaded_documents.append(document)
                # DIAGNOSTIC - Document chargé individuellement
                print(f"DIAGNOSTIC - Loaded: {document.metadata.filename}, {len(document.content)} chars")
                self.logger.info(
                    f"Chargé: {document.metadata.filename} "
                    f"({document.metadata.total_pages} pages, {document.metadata.custom_interviewee_name})"
                )
            except Exception as e:
                error_msg = f"Erreur lors du chargement de {pdf_file}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        if not loaded_documents:
            raise ValueError(f"Aucun document n'a pu être chargé. Erreurs: {errors}")
        
        if errors:
            self.logger.warning(
                f"{len(errors)} fichiers n'ont pas pu être chargés sur {len(pdf_files)} total"
            )
        
        # Validation du corpus (19 documents attendus)
        if len(loaded_documents) != 19:
            self.logger.warning(
                f"Attention: {len(loaded_documents)} documents chargés (19 attendus selon le cahier des charges)"
            )
        
        self.logger.info(f"Chargement terminé: {len(loaded_documents)} documents prêts")
        # DIAGNOSTIC - Vérification chargement PDF total
        print(f"DIAGNOSTIC - Documents loaded: {len(loaded_documents)}")
        if loaded_documents:
            total_chars = sum(len(doc.content) for doc in loaded_documents)
            print(f"DIAGNOSTIC - Total characters loaded: {total_chars}")
        return loaded_documents
    
    def load_single_pdf(self, file_path: str) -> LoadedDocument:
        """
        Charge un fichier PDF individuel
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            LoadedDocument: Document chargé avec ses métadonnées
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le PDF est corrompu ou illisible
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
        
        # Validation de la structure du PDF
        if not self.validate_pdf_structure(str(file_path)):
            raise ValueError(f"Le fichier {file_path} n'est pas un PDF valide")
        
        # Extraction du nombre de pages AVANT la création des métadonnées
        page_count = self._get_page_count(file_path)
        
        # Extraction des métadonnées avec le page_count
        metadata = self.extract_custom_metadata(str(file_path), page_count)
        
        # Extraction du contenu textuel
        try:
            content = self._extract_text_content(file_path)
            
        except Exception as e:
            raise ValueError(f"Erreur lors de l'extraction du contenu de {file_path}: {e}")
        
        # CORRECTION: LoadedDocument sans page_count et filename séparés
        return LoadedDocument(
            content=content,
            metadata=metadata
        )
    
    def extract_custom_metadata(self, file_path: str, page_count: int = 1) -> DocumentMetadata:
        """
        Extrait les métadonnées personnalisées d'un PDF
        
        CORRECTION: Ajout du paramètre page_count
        
        Les métadonnées sont extraites depuis:
        1. Le nom du fichier (format ELCA standardisé)
        2. Les métadonnées PDF natives (si disponibles)
        
        Args:
            file_path: Chemin vers le fichier PDF
            page_count: Nombre de pages du document
            
        Returns:
            DocumentMetadata: Métadonnées extraites du PDF
            
        Raises:
            ValueError: Si le format du nom de fichier est invalide
        """
        file_path = Path(file_path)
        filename = file_path.name
        
        # Extraction depuis le nom de fichier
        metadata_from_filename = self._extract_metadata_from_filename(filename)
        
        # Tentative d'extraction depuis les métadonnées PDF natives
        metadata_from_pdf = self._extract_metadata_from_pdf(file_path)
        
        # CORRECTION: Ajout de total_pages dans les métadonnées
        return DocumentMetadata(
            custom_interviewee_name=metadata_from_filename.get('interviewee_name', 'Unknown'),
            custom_interview_date=metadata_from_filename.get('interview_date', ''),
            custom_interview_language=metadata_from_filename.get('language', ''),
            custom_archive_type=metadata_from_pdf.get('archive_type', 'entretien'),
            filename=filename,
            file_path=str(file_path),
            total_pages=page_count  # CORRECTION: Ajout de total_pages
        )
    
    def validate_pdf_structure(self, file_path: str) -> bool:
        """
        Vérifie la structure et la validité d'un PDF
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            bool: True si le PDF est valide et accessible
        """
        try:
            file_path = Path(file_path)
            
            # Vérification de l'existence et de l'extension
            if not file_path.exists():
                self.logger.error(f"Fichier inexistant: {file_path}")
                return False
            
            if file_path.suffix.lower() not in {'.pdf'}:
                self.logger.error(f"Extension invalide: {file_path}")
                return False
            
            # Vérification de la taille (fichier non vide)
            if file_path.stat().st_size == 0:
                self.logger.error(f"Fichier vide: {file_path}")
                return False
            
            # Tentative d'ouverture avec PyPDF2
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Vérification du nombre de pages
                num_pages = len(pdf_reader.pages)
                if num_pages == 0:
                    self.logger.error(f"PDF sans pages: {file_path}")
                    return False
                
                # Tentative de lecture de la première page
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()
                
                # Validation minimale du contenu
                if not text or len(text.strip()) < 10:
                    self.logger.warning(
                        f"Contenu textuel très limité dans {file_path}"
                    )
                
            return True
            
        except PyPDF2.errors.PdfReadError as e:
            self.logger.error(f"Erreur de lecture PDF pour {file_path}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation de {file_path}: {e}")
            return False
    
    def get_pdf_files_list(self) -> List[str]:
        """
        Récupère la liste des fichiers PDF dans le dossier data/
        
        Returns:
            List[str]: Liste des chemins vers les fichiers PDF
        """
        if not self.data_directory.exists():
            return []
        
        pdf_files = []
        
        # Recherche récursive des fichiers PDF
        for file_path in self.data_directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                pdf_files.append(str(file_path))
        
        # Tri par nom pour un ordre cohérent
        pdf_files.sort()
        
        self.logger.debug(f"Fichiers PDF trouvés: {len(pdf_files)}")
        for pdf_file in pdf_files:
            self.logger.debug(f"  - {pdf_file}")
        
        return pdf_files
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extrait les métadonnées depuis le nom de fichier ELCA standardisé
        
        Format attendu: YYYYMMDD_ELCA_Nom_Prenom_entretien_LG[_page]
        Exemple: 20240627_ELCA_Vincent_Schaller_entretien_FR_1
        
        Args:
            filename: Nom du fichier
            
        Returns:
            Dict[str, str]: Métadonnées extraites
        """
        # Suppression de l'extension
        name_without_ext = Path(filename).stem
        
        # Application du pattern regex
        match = self.filename_pattern.match(name_without_ext)
        
        if not match:
            self.logger.warning(
                f"Format de nom de fichier non standard: {filename}"
            )
            # Tentative d'extraction fallback
            return self._fallback_metadata_extraction(filename)
        
        date_str, first_name, last_name, language_code, page = match.groups()
        
        # Construction du nom complet
        interviewee_name = f"{first_name} {last_name}"
        
        # Conversion de la date
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            formatted_date = date_obj.strftime('%d/%m/%Y')
        except ValueError:
            formatted_date = date_str
            self.logger.warning(f"Format de date invalide dans {filename}: {date_str}")
        
        # Conversion du code langue
        language = self.language_mapping.get(language_code, language_code.lower())
        
        return {
            'interviewee_name': interviewee_name,
            'interview_date': formatted_date,
            'language': language,
            'language_code': language_code,
            'raw_date': date_str
        }
    
    def _fallback_metadata_extraction(self, filename: str) -> Dict[str, str]:
        """
        Extraction de métadonnées en mode fallback pour fichiers non standard
        
        Args:
            filename: Nom du fichier
            
        Returns:
            Dict[str, str]: Métadonnées minimales extraites
        """
        name_without_ext = Path(filename).stem
        
        # Recherche de patterns connus
        interviewee_name = "Unknown"
        language = "français"  # Par défaut
        interview_date = ""
        
        # Recherche du nom (après ELCA_)
        if "ELCA_" in name_without_ext:
            parts = name_without_ext.split("_")
            if len(parts) >= 3:
                # Tentative de reconstruction du nom
                name_parts = []
                for i, part in enumerate(parts):
                    if part == "ELCA" and i + 1 < len(parts):
                        # Récupération des parties suivantes jusqu'à "entretien"
                        for j in range(i + 1, len(parts)):
                            if parts[j].lower() == "entretien":
                                break
                            name_parts.append(parts[j])
                        break
                
                if name_parts:
                    interviewee_name = " ".join(name_parts)
        
        # Recherche de la langue
        if "_AN" in name_without_ext or "_anglais" in name_without_ext:
            language = "anglais"
        
        self.logger.info(
            f"Extraction fallback pour {filename}: {interviewee_name}, {language}"
        )
        
        return {
            'interviewee_name': interviewee_name,
            'interview_date': interview_date,
            'language': language,
            'language_code': 'FR' if language == 'français' else 'AN'
        }
    
    def _extract_metadata_from_pdf(self, file_path: Path) -> Dict[str, str]:
        """
        Extrait les métadonnées depuis les métadonnées natives du PDF
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            Dict[str, str]: Métadonnées extraites du PDF
        """
        metadata = {}
        
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                if pdf_reader.metadata:
                    # Extraction des métadonnées personnalisées ELCA
                    pdf_metadata = pdf_reader.metadata
                    
                    # Recherche des champs personnalisés mentionnés dans le cahier des charges
                    custom_fields = {
                        'Customintervieweename': 'interviewee_name',
                        'Custominterviewdate': 'interview_date', 
                        'Custominterviewlanguage': 'language',
                        'Customarchivetype': 'archive_type'
                    }
                    
                    for pdf_key, internal_key in custom_fields.items():
                        if f'/{pdf_key}' in pdf_metadata:
                            metadata[internal_key] = str(pdf_metadata[f'/{pdf_key}'])
                        elif pdf_key in pdf_metadata:
                            metadata[internal_key] = str(pdf_metadata[pdf_key])
                    
                    # Métadonnées standard qui pourraient être utiles
                    standard_fields = {
                        '/Title': 'title',
                        '/Author': 'author',
                        '/Subject': 'subject',
                        '/Creator': 'creator',
                        '/CreationDate': 'creation_date'
                    }
                    
                    for pdf_key, internal_key in standard_fields.items():
                        if pdf_key in pdf_metadata:
                            metadata[internal_key] = str(pdf_metadata[pdf_key])
                
        except Exception as e:
            self.logger.debug(f"Impossible d'extraire les métadonnées PDF de {file_path}: {e}")
        
        return metadata
    
    def _extract_text_content(self, file_path: Path) -> str:
        """
        Extrait le contenu textuel d'un PDF SANS les métadonnées intégrées
        
        MODIFICATION: Suppression de l'enrichissement automatique des métadonnées
        Les métadonnées seront gérées séparément par le chunking
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            str: Contenu textuel pur du PDF
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Ajout d'un marqueur de page pour le retrieval
                            text_content.append(f"[PAGE {page_num}]\n{page_text}")
                        else:
                            self.logger.warning(
                                f"Page {page_num} de {file_path.name} ne contient pas de texte extractible"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Erreur lors de l'extraction de la page {page_num} "
                            f"de {file_path.name}: {e}"
                        )
                
                full_content = "\n\n".join(text_content)
                
                # Validation du contenu extrait
                if len(full_content.strip()) < 100:
                    self.logger.warning(
                        f"Contenu textuel très court pour {file_path.name}: "
                        f"{len(full_content)} caractères"
                    )
                
                return full_content  # Retour du contenu pur sans métadonnées
            
        except Exception as e:
            raise ValueError(f"Erreur lors de l'extraction du texte de {file_path}: {e}")

    def get_searchable_metadata_content(self, file_path: Path) -> str:
        """
        NOUVELLE MÉTHODE: Retourne le contenu des métadonnées sous forme recherchable
        
        Cette méthode sera utilisée par le chunking pour créer des chunks séparés
        pour les métadonnées.
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            str: Contenu des métadonnées formaté pour la recherche
        """
        metadata_from_pdf = self._extract_metadata_from_pdf(file_path)
        metadata_from_filename = self._extract_metadata_from_filename(file_path.name)
        
        return self._build_searchable_metadata_prefix(metadata_from_filename, metadata_from_pdf)

    def _build_searchable_metadata_prefix(self, filename_metadata: Dict[str, str], 
                                        pdf_metadata: Dict[str, str]) -> str:
        """
        NOUVELLE MÉTHODE : Construit un préfixe de métadonnées recherchables
        
        Args:
            filename_metadata: Métadonnées extraites du nom de fichier
            pdf_metadata: Métadonnées extraites du PDF
            
        Returns:
            str: Préfixe formaté pour la recherche
        """
        # Priorisation : métadonnées PDF > métadonnées filename > valeurs par défaut
        interviewee_name = (
            pdf_metadata.get('interviewee_name') or 
            filename_metadata.get('interviewee_name', 'Nom non spécifié')
        )
        
        interview_date = (
            pdf_metadata.get('interview_date') or 
            filename_metadata.get('interview_date', 'Date non spécifiée')
        )
        
        interview_language = (
            pdf_metadata.get('language') or 
            filename_metadata.get('language', 'Langue non spécifiée')
        )
        
        archive_type = pdf_metadata.get('archive_type', 'entretien')
        
        # Construction du préfixe de métadonnées recherchables
        metadata_prefix = f"""=== INFORMATIONS ENTRETIEN ===
    Nom de l'interviewé: {interviewee_name}
    Date de l'entretien: {interview_date}
    Langue de l'entretien: {interview_language}
    Type d'archive: {archive_type}
    Fichier source: {filename_metadata.get('filename', 'inconnu')}

    === CONTENU DE L'ENTRETIEN ==="""
        
        return metadata_prefix

    def _get_page_count(self, file_path: Path) -> int:
        """
        Compte le nombre de pages d'un PDF
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            int: Nombre de pages
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                return len(pdf_reader.pages)
        except Exception as e:
            self.logger.error(f"Erreur lors du comptage des pages de {file_path}: {e}")
            return 0
    
    def get_corpus_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du corpus chargé
        
        Returns:
            Dict[str, Any]: Statistiques détaillées
        """
        try:
            documents = self.load_all_pdfs()
            
            stats = {
                'total_documents': len(documents),
                'total_pages': sum(doc.metadata.total_pages for doc in documents),  # CORRECTION
                'languages': {},
                'interviewees': [],
                'average_pages_per_document': 0,
                'date_range': {'earliest': None, 'latest': None},
                'total_characters': 0
            }
            
            # Analyse par langue
            for doc in documents:
                lang = doc.metadata.custom_interview_language
                if lang in stats['languages']:
                    stats['languages'][lang] += 1
                else:
                    stats['languages'][lang] = 1
                
                # Liste des interviewés
                stats['interviewees'].append({
                    'name': doc.metadata.custom_interviewee_name,
                    'date': doc.metadata.custom_interview_date,
                    'language': lang,
                    'pages': doc.metadata.total_pages  # CORRECTION
                })
                
                # Comptage des caractères
                stats['total_characters'] += len(doc.content)
            
            # Calcul de la moyenne
            if documents:
                stats['average_pages_per_document'] = stats['total_pages'] / len(documents)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des statistiques: {e}")
            return {'error': str(e)}
