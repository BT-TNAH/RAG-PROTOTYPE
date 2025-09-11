from typing import List, Dict, Any, Optional
import logging
import re
import ollama
from dataclasses import dataclass
from config.configuration import LLMModel
from retrieval.retriever import RetrievalResult

@dataclass
class GenerationResponse:
    """
    Réponse générée par le LLM avec métadonnées
    """
    answer: str
    sources_used: List[str]
    model_used: str
    confidence_score: Optional[float] = None
    generation_time: Optional[float] = None

class LLMGenerator:
    """
    Générateur de réponses utilisant Ollama pour les LLM locaux
    
    Ce module gère :
    - L'interface avec Ollama pour la génération
    - La création des prompts système et utilisateur
    - L'extraction et validation des sources citées
    - La gestion des erreurs de génération
    """
    
    def __init__(self, llm_model):
        """
        Initialise le générateur avec le modèle LLM spécifié
    
        Args:
            llm_model: Configuration du modèle LLM (string ou objet LLMModel)
        """
        # Gestion robuste des différents types d'entrée
        if isinstance(llm_model, str):
            self.model_name = llm_model
        elif hasattr(llm_model, 'model_name'):
            self.model_name = llm_model.model_name
        elif hasattr(llm_model, 'value'):  # Pour les enums
            self.model_name = llm_model.value
        elif hasattr(llm_model, 'name'):
            self.model_name = llm_model.name
        else:
            self.model_name = str(llm_model)
    
        self.ollama_client = None
        self.logger = logging.getLogger(__name__)
    
        # Configuration des paramètres de génération
        self.generation_params = {
            'temperature': 0.1,  # Faible pour plus de cohérence
            'top_p': 0.9,
            'top_k': 40,
            'num_predict': 1000,  # Limite de tokens générés
            'stop': ['<END>', '---', 'Human:', 'Assistant:']
        }
        self._last_cited_sources = []
    
    def initialize_ollama_client(self) -> bool:
        """
        Initialise le client Ollama pour la génération
        
        Returns:
            bool: True si l'initialisation réussit, False sinon
        """
        try:
            # Test de connexion à Ollama
            response = ollama.list()
            
            # CORRECTION 1: Accès correct aux modèles
            models_data = response.get('models', [])
            available_models = []
            
            for model in models_data:
                # Gestion robuste des différents formats de réponse d'Ollama
                if isinstance(model, dict):
                    model_name = model.get('name', model.get('model', ''))
                else:
                    model_name = str(model)
                
                if model_name:
                    available_models.append(model_name)
            
            # CORRECTION 2: Vérification plus flexible du nom du modèle
            model_found = False
            for available_model in available_models:
                if self.model_name in available_model or available_model.startswith(self.model_name.split(':')[0]):
                    model_found = True
                    break
            
            if not model_found:
                self.logger.error(f"Modèle {self.model_name} non disponible dans Ollama")
                self.logger.info(f"Modèles disponibles : {available_models}")
                return False
            
            self.ollama_client = ollama
            self.logger.info(f"Client Ollama initialisé avec le modèle {self.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation d'Ollama : {e}")
            return False

    def generate_response(self, query: str, chunks_or_context, retrieval_results: Optional[List[RetrievalResult]] = None) -> str:
        """
        Génère une réponse basée sur la requête et le contexte
        
        Args:
            query: Question de l'utilisateur
            context: Contexte récupéré par le retrieval
            retrieval_results: Résultats du retrieval pour les citations
            
        Returns:
            str: Réponse générée (string directe au lieu de GenerationResponse)
        """
        try:
            # Validation des entrées avec type checking
            if not isinstance(query, str):
                query = str(query) if query is not None else ""

            if not query.strip():
                return "Erreur : La requête ne peut pas être vide"
            
            # Gestion des deux types d'entrée avec validation de type
            if isinstance(chunks_or_context, str):
                context = chunks_or_context
                available_sources = []
            elif chunks_or_context is None:
                context = "Aucun contexte fourni"
                available_sources = []
            else:
                context = self._chunks_to_context(chunks_or_context)
                retrieval_results = retrieval_results or []
                available_sources = self._extract_available_sources(chunks_or_context)

            if not context.strip():
                self.logger.warning("Contexte vide - génération sans contexte")
                context = "Aucun contexte pertinent trouvé dans les documents."
            
            # CORRECTION 3: Initialisation sécurisée avec retry
            max_retries = 3
            for attempt in range(max_retries):
                if not self.ollama_client:
                    if not self.initialize_ollama_client():
                        if attempt == max_retries - 1:
                            return "Erreur : Impossible d'initialiser le client Ollama après plusieurs tentatives"
                        continue
                break
            
            # Création des prompts
            system_prompt = self.create_system_prompt()
            user_prompt = self.create_user_prompt(query, context)
            
            self.logger.debug(f"Génération avec le modèle {self.model_name}")
            
            # CORRECTION 4: Gestion d'erreur robuste pour la génération
            try:
                response = self.ollama_client.generate(
                    model=self.model_name,
                    system=system_prompt,
                    prompt=user_prompt,
                    options=self.generation_params
                )
            except Exception as ollama_error:
                self.logger.error(f"Erreur spécifique Ollama : {ollama_error}")
                # Tentative avec des paramètres simplifiés
                try:
                    response = self.ollama_client.generate(
                        model=self.model_name,
                        prompt=f"{system_prompt}\n\n{user_prompt}"
                    )
                except Exception as fallback_error:
                    self.logger.error(f"Erreur même en fallback : {fallback_error}")
                    return f"Erreur de génération : {str(fallback_error)}"
            
            # Extraction de la réponse
            generated_answer = response.get('response', '').strip()
            
            if not generated_answer:
                generated_answer = "Je n'ai pas pu générer une réponse appropriée pour cette question."
            
            # Post-traitement de la réponse
            generated_answer = self._post_process_response(generated_answer)
            
            self.logger.info(f"Réponse générée avec succès")
            # Extraction des sources citées pour les colonnes K et L
            cited_sources = self.extract_cited_sources_from_response(generated_answer)

            # Stocker les sources citées comme attribut pour récupération ultérieure
            self._last_cited_sources = cited_sources

            self.logger.info(f"Réponse générée avec succès - {len(cited_sources)} sources citées uniques")
            return generated_answer  # Retour direct de la string
            
        except Exception as e:
            error_message = f"Erreur lors de la génération de la réponse : {str(e)}"
            self.logger.error(error_message)
            return error_message
    
    def create_system_prompt(self) -> str:
        """
        Créé le prompt système pour le LLM avec OBLIGATION de citer les sources
        
        Returns:
            str: Prompt système optimisé pour forcer les citations
        """
        system_prompt = """Tu es un assistant spécialisé dans l'analyse d'archives orales d'entreprise. 

    RÔLE ET CONTEXTE :
    - Tu analyses des transcriptions d'entretiens sur l'histoire de l'entreprise ELCA
    - Tu réponds uniquement en te basant sur les informations fournies dans le contexte
    - Tu cites OBLIGATOIREMENT tes sources avec le format [nom_fichier_page] ou [nom_fichier_meta]

    RÈGLES STRICTES DE CITATION :
    1. CITATION OBLIGATOIRE : Tu DOIS citer au format [fichier_page] pour CHAQUE information que tu donnes
    2. AUCUNE INFORMATION SANS SOURCE : Si tu mentionnes un fait, il DOIT être suivi de sa citation
    3. SOURCES MULTIPLES : Si plusieurs sources supportent la même information, cite-les toutes : [source1, source2]
    4. INFORMATIONS ABSENTES : Si l'information n'est pas dans le contexte, dis "Cette information n'est pas présente dans les documents fournis" - SANS inventer de sources

    INSTRUCTIONS SPÉCIFIQUES :
    1. FIDÉLITÉ AU CONTEXTE : Ne réponds QUE avec les informations présentes dans le contexte fourni
    2. CITATIONS SYSTÉMATIQUES : Chaque phrase contenant une information doit avoir sa citation
    3. FORMAT DES CITATIONS : [fichier_page] ou [fichier_meta] selon le type de source
    4. MÉTADONNÉES : Utilise les métadonnées des fichiers (nom de l'interviewé, date, langue) avec citation [fichier_meta]
    5. PRÉCISION : Sois précis et factuel, évite les interprétations sans sources

    FORMAT DES CITATIONS OBLIGATOIRES :
    - Une information : "John Doe a rejoint ELCA en 1995 [20240611_ELCA_Hakan_Birsel_entretien_FR_5]."
    - Plusieurs sources : "Cette stratégie était commune [20240611_ELCA_Hakan_Birsel_entretien_FR_5, 20240626_ELCA_Alain_Berguerand_entretien_FR_3]."
    - Métadonnées : "L'entretien s'est déroulé en français [20240611_ELCA_Hakan_Birsel_entretien_FR_meta]."

    STRUCTURE DE RÉPONSE OBLIGATOIRE :
    - Commence directement par la réponse avec citations
    - Chaque affirmation = une citation immédiate
    - Si pas de sources pertinentes = dis-le clairement et ne cite rien
    - N'ajoute pas de récapitulatif final des sources

    EXEMPLES DE RÉPONSES CORRECTES :
    ✓ "L'entreprise a été fondée en 1989 [document1_meta]. La croissance initiale s'est faite par acquisition [document1_5, document2_3]."
    ✗ "L'entreprise a été fondée en 1989 et a grandi rapidement." (MANQUE LES CITATIONS)

    RAPPEL CRITIQUE : Tu ne peux citer QUE les sources présentes dans le contexte fourni. Jamais d'invention."""

        return system_prompt
    
    def create_user_prompt(self, query: str, context: str) -> str:
        """
        Créé le prompt utilisateur avec la requête et le contexte
        RENFORCÉ pour obliger les citations
        
        Args:
            query: Question utilisateur
            context: Contexte récupéré des documents
            
        Returns:
            str: Prompt utilisateur complet avec instructions strictes
        """
        # Nettoyage et structuration du contexte
        context_parts = context.split('\n\n') if context else []
        formatted_context = ""
        
        for i, part in enumerate(context_parts, 1):
            if part.strip():
                formatted_context += f"DOCUMENT {i}:\n{part.strip()}\n\n"
        
        if not formatted_context:
            formatted_context = "CONTEXTE : Aucun document pertinent trouvé.\n\n"
        
        # Ajout de la liste des sources autorisées
        sources_info = ""
        if hasattr(self, '_current_available_sources') and self._current_available_sources:
            sources_info = f"\nSOURCES AUTORISÉES POUR CITATIONS :\n"
            for source in self._current_available_sources:
                sources_info += f"- {source}\n"
            sources_info += "\nIMPORTANT : Tu ne peux citer QUE ces sources exactes. Ne jamais inventer de références.\n"

        user_prompt = f"""CONTEXTE DOCUMENTAIRE :
    {formatted_context}
    {sources_info}
    QUESTION :
    {query}

    INSTRUCTIONS DE RÉPONSE STRICTES :
    1. Tu DOIS citer [source] pour chaque information que tu donnes
    2. Si aucun document ne répond à la question, dis-le clairement SANS citer de sources, en indiquant que c'est tu n'as pas trouvé de sources
    3. Utilise UNIQUEMENT les sources listées ci-dessus
    4. Format obligatoire : "Information précise [source1, source2]."

    RÉPONSE (avec citations OBLIGATOIRES au format [fichier_page] pour chaque affirmation, SANS récapitulatif final) :"""
        
        return user_prompt
    
    # CORRECTION 5: Méthode de validation améliorée
    def validate_llm_model(self) -> bool:
        """
        Vérifie que le modèle LLM est disponible dans Ollama
    
        Returns:
            bool: True si le modèle est disponible
        """
        try:
            response = ollama.list()
            models_data = response.get('models', [])
            available_models = []
        
            for model in models_data:
                if isinstance(model, dict):
                    model_name = model.get('name', model.get('model', ''))
                else:
                    model_name = str(model)
            
                if model_name:
                    available_models.append(model_name)
        
            # Vérification flexible
            model_found = any(
                self.model_name in model or model.startswith(self.model_name.split(':')[0])
                for model in available_models
            )
        
            if model_found:
                self.logger.info(f"Modèle {self.model_name} validé et disponible")
            else:
                self.logger.error(f"Modèle {self.model_name} non disponible")
                self.logger.info(f"Modèles disponibles : {available_models}")
        
            return model_found
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation du modèle : {e}")
            return False
    
    def extract_sources_from_response(self, response: str, retrieval_results: List[RetrievalResult]) -> List[str]:
        """
        Extrait et valide les sources citées dans la réponse
        
        CORRIGÉ : Validation stricte basée sur source_reference du retrieval
        """
        try:
            # Extraction des citations au format [fichier_page] ou [fichier_meta]
            citation_pattern = r'\[([^\]]+)\]'
            citations = re.findall(citation_pattern, response)
            
            # Validation des citations
            valid_sources = []
            available_sources = set()
            
            # NOUVEAU : Construction exacte des sources disponibles
            for result in retrieval_results:
                # Utilisation DIRECTE de la source_reference du retrieval
                if hasattr(result, 'source_reference') and result.source_reference:
                    available_sources.add(result.source_reference)
                else:
                    # Fallback seulement si source_reference manque
                    self.logger.warning(f"source_reference manquant pour un résultat")
                    # Génération manuelle comme fallback
                    if hasattr(result, 'chunk') and result.chunk.metadata:
                        metadata = result.chunk.metadata
                        is_metadata_chunk = metadata.get('is_metadata', False)
                        
                        source = metadata.get('source', '') or metadata.get('filename', '')
                        if source.lower().endswith('.pdf'):
                            source = source[:-4]
                        
                        if is_metadata_chunk:
                            available_sources.add(f"{source}_meta")
                        else:
                            page = metadata.get('page', '') or metadata.get('page_number', '')
                            if source and page:
                                available_sources.add(f"{source}_{page}")

            self.logger.debug(f"Sources disponibles pour validation : {available_sources}")
            
            # Validation stricte - correspondance exacte requise
            for citation in citations:
                citation = citation.strip()
                
                if citation in available_sources:
                    if citation not in valid_sources:
                        valid_sources.append(citation)
                else:
                    self.logger.warning(f"Source citée NON VALIDÉE (correspondance exacte requise) : '{citation}'")
                    self.logger.debug(f"Sources disponibles : {available_sources}")
            
            # Fallback si aucune source validée mais des résultats existent
            if not valid_sources and retrieval_results:
                self.logger.warning("Aucune source citée validée - utilisation des 3 premières sources du retrieval")
                for result in retrieval_results[:3]:
                    if hasattr(result, 'source_reference') and result.source_reference:
                        if result.source_reference not in valid_sources:
                            valid_sources.append(result.source_reference)
            
            self.logger.debug(f"Sources extraites et validées : {valid_sources}")
            return valid_sources
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction des sources : {e}")
            return []
        
    def extract_cited_sources_from_response(self, response_text: str) -> List[str]:
        """
        Extrait les sources citées dans la réponse générée sans doublons
        
        NOUVEAU : Méthode dédiée aux sources citées dans la réponse générée
        Utilise la même logique que extract_sources_from_response() mais
        retourne une liste unique des sources citées (sans répétitions)
        
        Args:
            response_text: Texte de la réponse générée par le LLM
            
        Returns:
            List[str]: Liste unique des sources citées dans la réponse
        """
        try:
            # Extraction des citations au format [fichier_page] ou [fichier_meta]
            citation_pattern = r'\[([^\]]+)\]'
            citations = re.findall(citation_pattern, response_text)
            
            if not citations:
                self.logger.debug("Aucune citation trouvée dans la réponse générée")
                return []
            
            # Suppression des doublons tout en préservant l'ordre
            unique_cited_sources = []
            seen = set()
            
            for citation in citations:
                citation = citation.strip()
                if citation and citation not in seen:
                    unique_cited_sources.append(citation)
                    seen.add(citation)
            
            self.logger.debug(f"Sources citées uniques extraites : {unique_cited_sources}")
            return unique_cited_sources
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction des sources citées : {e}")
            return []   
    
    def _post_process_response(self, response: str) -> str:
        """
        Post-traitement de la réponse générée
        
        Args:
            response: Réponse brute du LLM
            
        Returns:
            str: Réponse nettoyée et formatée
        """
        # Suppression des artefacts de génération
        response = re.sub(r'\n\s*\n', '\n\n', response)  # Lignes vides multiples
        response = response.strip()
        
        # Suppression des marqueurs de fin potentiels
        end_markers = ['<END>', '---', 'Human:', 'Assistant:']
        for marker in end_markers:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        # Nettoyage des espaces
        response = '\n'.join(line.strip() for line in response.split('\n'))
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle configuré
        
        Returns:
            Dict[str, Any]: Informations du modèle
        """
        return {
            'model_name': self.model_name,
            'model_type': 'LLM',
            'is_initialized': self.ollama_client is not None,
            'generation_params': self.generation_params
        }
    
    def update_generation_params(self, **kwargs) -> None:
        """
        Met à jour les paramètres de génération
        
        Args:
            **kwargs: Nouveaux paramètres de génération
        """
        self.generation_params.update(kwargs)
        self.logger.info(f"Paramètres de génération mis à jour : {kwargs}")

    def _chunks_to_context(self, chunks: List[Any]) -> str:
        """Convertit une liste de chunks en contexte string"""
        context_parts = []
        for chunk in chunks:
            if hasattr(chunk, 'content'):
                context_parts.append(chunk.content)
            elif hasattr(chunk, 'text'):
                context_parts.append(chunk.text)
            else:
                context_parts.append(str(chunk))
        return "\n\n".join(context_parts)

    def _extract_available_sources(self, chunks: List[Any]) -> List[str]:
        """
        Extrait les sources disponibles des chunks pour les citations
        
        NOUVEAU : Utilise source_reference du retrieval ou génère avec _meta/_page
        """
        sources = []
        for chunk in chunks:
            # NOUVEAU : Priorité à source_reference si disponible (déjà formaté par le retrieval)
            if hasattr(chunk, 'source_reference') and chunk.source_reference:
                if chunk.source_reference not in sources:
                    sources.append(chunk.source_reference)
                continue
            
            # Fallback : génération manuelle de la référence
            if hasattr(chunk, 'metadata') and chunk.metadata:
                is_metadata_chunk = chunk.metadata.get('is_metadata', False)
                source = chunk.metadata.get('source', '') or chunk.metadata.get('filename', '')
                
                if source.lower().endswith('.pdf'):
                    source = source[:-4]
                
                if is_metadata_chunk:
                    # NOUVEAU : Chunk de métadonnées
                    source_ref = f"{source}_meta"
                else:
                    # Chunk de contenu normal
                    page = chunk.metadata.get('page', '') or chunk.metadata.get('page_number', '')
                    source_ref = f"{source}_{page}" if page else source
                
                if source_ref not in sources:
                    sources.append(source_ref)
        
        # Stockage temporaire pour create_user_prompt
        self._current_available_sources = sources
        return sources
    
    def get_last_cited_sources(self) -> List[str]:
        """
        Récupère les sources citées de la dernière génération
        
        NOUVEAU : Permet de récupérer les sources citées après la génération
        pour les colonnes K et L de l'évaluation
        
        Returns:
            List[str]: Sources citées dans la dernière réponse générée
        """
        return getattr(self, '_last_cited_sources', [])

    def format_cited_sources_for_csv(self, cited_sources: List[str]) -> str:
        """
        Formate les sources citées pour l'export CSV (colonne K)
        
        NOUVEAU : Formatage spécifique pour les sources citées
        
        Args:
            cited_sources: Liste des sources citées
            
        Returns:
            str: Sources formatées pour CSV "[source1, source2, ...]"
        """
        if not cited_sources:
            return "[]"
        
        try:
            # Formatage identique aux sources trouvées par retrieval
            sources_str = ", ".join(cited_sources)
            return f"[{sources_str}]"
            
        except Exception as e:
            self.logger.error(f"Erreur lors du formatage des sources citées : {e}")
            return "[]"
