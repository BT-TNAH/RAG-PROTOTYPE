from typing import Optional, List, Dict, Any
import sys
import logging
from pathlib import Path
from datetime import datetime

# Import des classes de l'architecture (cohérentes avec main.py)
from retrieval.retriever import DocumentRetriever, RetrievalResult
from generation.generator import LLMGenerator as ResponseGenerator


class ChatbotMode:
    """
    Mode chatbot interactif pour questions libres sur le corpus d'archives orales
    
    Permet à l'utilisateur de poser des questions libres en mode interactif
    avec affichage des réponses et citation des sources (fichier, page, passage).
    """
    
    def __init__(self, retriever: Optional[DocumentRetriever] = None, 
                 generator: Optional[ResponseGenerator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialise le mode chatbot
        
        Args:
            retriever: Retrieveur de documents (optionnel si fourni par main.py)
            generator: Générateur de réponses (optionnel si fourni par main.py)
            config: Configuration RAG (optionnelle si fournie par main.py)
        """
        self.retriever = retriever
        self.generator = generator
        self.current_config = config
        self.session_start_time = datetime.now()
        
        # Configuration du logging
        self.logger = logging.getLogger(__name__)
        
        # Statistiques de session
        self.questions_count = 0
        self.session_active = False
    
    def set_components(self, retriever: DocumentRetriever = None, 
                      generator: ResponseGenerator = None,
                      config: Dict[str, Any] = None) -> None:
        """
        Méthode pour assigner les composants après initialisation
        (compatibilité avec main.py)
        
        Args:
            retriever: Retrieveur de documents
            generator: Générateur de réponses
            config: Configuration RAG
        """
        if retriever:
            self.retriever = retriever
        if generator:
            self.generator = generator
        if config:
            self.current_config = config
    
    def start_interactive_session(self) -> None:
        """
        Démarre la session interactive de chat
        
        Appelée depuis main.py après initialisation des composants
        """
        try:
            # Vérification que les composants sont prêts
            if not self._components_ready():
                self.logger.error("Composants RAG non initialisés")
                print("❌ Erreur : Composants RAG non initialisés")
                return
                
            # Message d'accueil
            self.display_welcome_message()
            
            # Lancement de la boucle interactive
            self.interactive_chat_loop()
            
        except Exception as e:
            self.logger.error(f"Erreur dans start_interactive_session: {e}")
            print(f"❌ Erreur dans la session interactive : {e}")
    
    def interactive_chat_loop(self) -> None:
        """
        Boucle principale de chat interactif
        
        Gère :
        - Saisie des questions utilisateur
        - Traitement des commandes spéciales
        - Affichage des réponses avec sources
        - Gestion de l'arrêt de session
        """
        self.session_active = True
        
        try:
            while self.session_active:
                # Saisie de la question
                user_input = self._get_user_input()
                
                # Vérification des commandes spéciales
                if self.handle_special_commands(user_input):
                    continue
                
                # Traitement de la question
                if user_input.strip():
                    self.process_user_query(user_input.strip())
                else:
                    print("⚠️ Veuillez saisir une question non vide.")
                    
        except KeyboardInterrupt:
            print("\n\n👋 Session interrompue par l'utilisateur.")
            self._display_session_summary()
        except Exception as e:
            self.logger.error(f"Erreur dans la boucle interactive : {e}")
            print(f"❌ Erreur dans la session : {e}")
        finally:
            self.session_active = False
    
    def process_user_query(self, query: str) -> None:
        """
        Traite une requête utilisateur et affiche la réponse
        
        Args:
            query: Question de l'utilisateur
        """
        try:
            print(f"\n🤔 Traitement de votre question...")
            self.questions_count += 1
            
            # Étape 1 : Retrieval - Recherche des chunks pertinents
            print("🔍 Recherche dans le corpus...")
            retrieval_results = self.retriever.retrieve(query)
            
            if not retrieval_results:
                print("❌ Aucun document pertinent trouvé pour cette question.")
                print("💡 Essayez de reformuler votre question ou d'utiliser d'autres mots-clés.")
                return
            
            # CORRECTION: Extraction des chunks pour la génération
            chunks_for_generation = []
            for result in retrieval_results:
                if isinstance(result, RetrievalResult):
                    chunks_for_generation.append(result.chunk)
                else:
                    # Fallback pour les anciens formats
                    chunks_for_generation.append(result)
            
            # Étape 2 : Génération - Création de la réponse avec le LLM
            print("🧠 Génération de la réponse...")
            response = self.generator.generate_response(query, chunks_for_generation, retrieval_results=retrieval_results)
            
            # Étape 3 : Extraction des sources
            sources = self._extract_sources_from_results(retrieval_results)
            
            # Étape 4 : Affichage de la réponse avec sources
            self.display_response_with_sources(response, sources, retrieval_results)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de la requête '{query}': {e}")
            print(f"❌ Erreur lors du traitement de votre question : {e}")
            print("💡 Veuillez réessayer ou reformuler votre question.")
    
    def display_response_with_sources(self, response: str, sources: List[str], 
                                    retrieval_results: List) -> None:
        """
        Affiche la réponse avec les sources citées
        
        Args:
            response: Réponse générée par le LLM
            sources: Liste des sources formatées (fichier_page)
            retrieval_results: Résultats détaillés du retrieval pour extraits
        """
        print("\n" + "="*80)
        print("📝 RÉPONSE")
        print("="*80)
        print(response)
        print()
        
        print("📚 SOURCES CONSULTÉES")
        print("-"*80)
        
        if sources:
            for i, source in enumerate(sources, 1):
                # Récupération des métadonnées et extrait pour cette source
                result = retrieval_results[i-1] if i-1 < len(retrieval_results) else None
                
                print(f"[{i}] {source}")
                
                # CORRECTION: Gestion robuste des types RetrievalResult et fallback
                chunk = None
                metadata = {}
                
                if isinstance(result, RetrievalResult):
                    chunk = result.chunk
                    metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
                elif hasattr(result, 'metadata'):
                    chunk = result
                    metadata = result.metadata
                else:
                    # Fallback minimal
                    chunk = result
                    metadata = {}
                
                # Affichage des métadonnées si disponibles
                if metadata:
                    # CORRECTION: Utiliser les bons noms de métadonnées (cohérent avec evaluation_mode)
                    if 'custom_interviewee_name' in metadata:
                        print(f"    👤 Interviewé(e) : {metadata['custom_interviewee_name']}")
                    elif 'customintervieweename' in metadata:
                        print(f"    👤 Interviewé(e) : {metadata['customintervieweename']}")
                    elif 'interviewee_name' in metadata:
                        print(f"    👤 Interviewé(e) : {metadata['interviewee_name']}")
                    
                    if 'custom_interview_date' in metadata:
                        print(f"    📅 Date : {metadata['custom_interview_date']}")
                    elif 'custominterviewdate' in metadata:
                        print(f"    📅 Date : {metadata['custominterviewdate']}")
                    elif 'interview_date' in metadata:
                        print(f"    📅 Date : {metadata['interview_date']}")
                    
                    if 'custom_interview_language' in metadata:
                        print(f"    🌐 Langue : {metadata['custom_interview_language']}")
                    elif 'custominterviewlanguage' in metadata:
                        print(f"    🌐 Langue : {metadata['custominterviewlanguage']}")
                    elif 'interview_language' in metadata:
                        print(f"    🌐 Langue : {metadata['interview_language']}")
                
                # Affichage de l'extrait pertinent
                extract_text = None
                
                if chunk and hasattr(chunk, 'content'):
                    extract_text = chunk.content
                elif result and hasattr(result, 'content'):
                    extract_text = result.content
                elif result and hasattr(result, 'text'):
                    extract_text = result.text
                
                if extract_text:
                    # Limitation de l'extrait à 200 caractères
                    extract = extract_text[:200] + "..." if len(extract_text) > 200 else extract_text
                    print(f"    📄 Extrait : {extract}")
                
                print()
        else:
            print("Aucune source spécifique identifiée.")
        
        print("="*80)
        print(f"⏱️ Question {self.questions_count} traitée | Session active depuis {self._get_session_duration()}")
        print()
    
    def display_welcome_message(self) -> None:
        """
        Affiche le message d'accueil du chatbot
        """
        print("\n" + "🤖"*35)
        print("🤖" + " "*21 + "MODE CHATBOT INTERACTIF" + " "*22 + "🤖")
        print("🤖" + " "*15 + "Archives Orales XXXX - Questions Libres" + " "*12 + "🤖")
        print("🤖"*35)
        print()
        print("💬 Bienvenue dans le mode chatbot interactif !")
        print("📚 Vous pouvez maintenant poser des questions libres sur le corpus de 19 transcriptions.")
        print()
        
        # Affichage de la configuration actuelle
        if self.current_config:
            print("⚙️ Configuration actuelle :")
            # CORRECTION: Extraction cohérente des valeurs de configuration
            llm_model = self._extract_config_value(self.current_config.get('llm_model', 'Non défini'))
            embedding_model = self._extract_config_value(self.current_config.get('embedding_model', 'Non défini'))
            
            print(f"   • Modèle LLM : {llm_model}")
            print(f"   • Chunk Size : {self.current_config.get('chunk_size', 'Non défini')}")
            print(f"   • Chunk Overlap : {self.current_config.get('chunk_overlap', 'Non défini')}")
            print(f"   • Modèle Embedding : {embedding_model}")
            print(f"   • Top-K : {self.current_config.get('top_k', 'Non défini')}")
            print()
        
        print("💡 Conseils d'utilisation :")
        print("   • Posez des questions précises sur les entretiens")
        print("   • Mentionnez des noms, dates ou sujets spécifiques")
        print("   • Les réponses incluront les sources avec extraits")
        print()
        
        self.display_help_commands()
    
    def display_help_commands(self) -> None:
        """
        Affiche les commandes d'aide disponibles
        """
        print("🆘 Commandes spéciales disponibles :")
        print("   • '/aide' ou '/help'     → Afficher cette aide")
        print("   • '/config'             → Afficher la configuration actuelle")
        print("   • '/stats'              → Afficher les statistiques de session")
        print("   • '/quit' ou '/exit'    → Quitter le mode chatbot")
        print("   • '/clear'              → Effacer l'écran")
        print()
        print("🔍 Tapez votre question et appuyez sur Entrée pour commencer !")
        print("-"*80)
    
    def handle_special_commands(self, user_input: str) -> bool:
        """
        Gère les commandes spéciales (aide, quitter, etc.)
        
        Args:
            user_input: Input utilisateur
            
        Returns:
            bool: True si c'était une commande spéciale
        """
        command = user_input.lower().strip()
        
        if command in ['/aide', '/help']:
            self.display_help_commands()
            return True
        
        elif command == '/config':
            self._display_current_config()
            return True
        
        elif command == '/stats':
            self._display_session_stats()
            return True
        
        elif command in ['/quit', '/exit']:
            print("👋 Merci d'avoir utilisé le chatbot ELCA !")
            self._display_session_summary()
            self.session_active = False
            return True
        
        elif command == '/clear':
            self._clear_screen()
            return True
        
        # Pas une commande spéciale
        return False
    
    def _components_ready(self) -> bool:
        """
        Vérifie si les composants RAG sont prêts
        
        Returns:
            bool: True si retriever et generator sont initialisés
        """
        components_status = {
            'retriever': self.retriever is not None,
            'generator': self.generator is not None,
            'config': self.current_config is not None
        }
        
        # Log des composants manquants pour debug
        missing_components = [name for name, status in components_status.items() if not status]
        if missing_components:
            self.logger.error(f"Composants manquants dans ChatbotMode : {missing_components}")
        
        return all(components_status.values())
    
    def _get_user_input(self) -> str:
        """
        Récupère l'input utilisateur avec gestion des erreurs
        
        Returns:
            str: Question saisie par l'utilisateur
        """
        try:
            return input("💬 Votre question ➤ ")
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt()
    
    def _extract_sources_from_results(self, retrieval_results: List) -> List[str]:
        """
        Extrait les sources formatées depuis les résultats de retrieval
        
        Args:
            retrieval_results: Résultats du retrieval
            
        Returns:
            List[str]: Liste des sources au format "fichier_page"
        """
        sources = []
        
        for result in retrieval_results:
            try:
                source_ref = "source_inconnue"
                
                if isinstance(result, RetrievalResult):
                    # Utiliser la référence source pré-formatée
                    source_ref = result.source_reference
                else:
                    # Extraction manuelle depuis les métadonnées
                    if hasattr(result, 'metadata') and result.metadata:
                        filename = result.metadata.get('filename', 'fichier_inconnu')
                        page = result.metadata.get('page', result.metadata.get('page_number', 'page_inconnue'))
                        
                        # Nettoyage du nom de fichier (enlever l'extension)
                        if filename.endswith('.pdf'):
                            filename = filename[:-4]
                        
                        source_ref = f"{filename}_page{page}"
                
                sources.append(source_ref)
                    
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'extraction d'une source : {e}")
                sources.append("source_erreur")
        
        return sources
    
    def _extract_config_value(self, config_value) -> str:
        """
        Extrait la valeur de configuration (cohérent avec main.py)
        
        Args:
            config_value: Valeur de config (peut être string, enum, objet)
            
        Returns:
            str: Valeur formatée
        """
        if isinstance(config_value, str):
            return config_value
        elif hasattr(config_value, 'value'):
            return config_value.value
        elif hasattr(config_value, 'name'):
            return config_value.name
        else:
            return str(config_value)
    
    def _display_current_config(self) -> None:
        """
        Affiche la configuration actuelle détaillée
        """
        print("\n⚙️ CONFIGURATION ACTUELLE")
        print("-"*40)
        
        if self.current_config:
            for key, value in self.current_config.items():
                # Extraction cohérente des valeurs
                display_value = self._extract_config_value(value)
                print(f"   {key} : {display_value}")
        else:
            print("   Aucune configuration disponible")
        
        print("-"*40)
        print()
    
    def _display_session_stats(self) -> None:
        """
        Affiche les statistiques de la session en cours
        """
        duration = self._get_session_duration()
        
        print("\n📊 STATISTIQUES DE SESSION")
        print("-"*40)
        print(f"   Démarrage : {self.session_start_time.strftime('%H:%M:%S')}")
        print(f"   Durée : {duration}")
        print(f"   Questions posées : {self.questions_count}")
        
        if self.questions_count > 0:
            avg_time = (datetime.now() - self.session_start_time).total_seconds() / self.questions_count
            print(f"   Temps moyen/question : {avg_time:.1f}s")
        
        print("-"*40)
        print()
    
    def _display_session_summary(self) -> None:
        """
        Affiche un résumé de la session à la fin
        """
        duration = self._get_session_duration()
        
        print("\n📋 RÉSUMÉ DE SESSION")
        print("="*40)
        print(f"Durée totale : {duration}")
        print(f"Questions posées : {self.questions_count}")
        print("Merci d'avoir utilisé le chatbot ELCA ! 👋")
        print("="*40)
    
    def _get_session_duration(self) -> str:
        """
        Calcule et formate la durée de la session
        
        Returns:
            str: Durée formatée (ex: "5m 32s")
        """
        delta = datetime.now() - self.session_start_time
        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _clear_screen(self) -> None:
        """
        Efface l'écran du terminal
        """
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Réaffichage du header après clear
        print("🤖 MODE CHATBOT INTERACTIF - ELCA")
        print("-"*40)
