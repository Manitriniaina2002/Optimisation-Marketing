import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import io
import os
from typing import Union, Dict, List, Optional, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Classe pour charger et gérer les données de l'application d'analyse marketing.
    """
    
    def __init__(self, upload_dir: Union[str, Path] = "data/uploads"):
        """
        Initialise le chargeur de données avec le répertoire de téléversement.
        
        Args:
            upload_dir: Chemin vers le répertoire de téléversement
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}
        self.metadata = {}
        
    def save_uploaded_file(self, uploaded_file, subfolder: str = "") -> Path:
        """
        Enregistre un fichier téléversé dans le répertoire d'upload.
        
        Args:
            uploaded_file: Fichier téléversé via st.file_uploader
            subfolder: Sous-dossier pour organiser les fichiers
            
        Returns:
            Path: Chemin vers le fichier enregistré
        """
        try:
            # Créer le sous-dossier si nécessaire
            save_dir = self.upload_dir / subfolder
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Construire le chemin de destination
            file_path = save_dir / uploaded_file.name
            
            # Enregistrer le fichier
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            logger.info(f"Fichier enregistré : {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du fichier : {e}")
            raise
    
    def load_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Charge un fichier CSV dans un DataFrame pandas.
        
        Args:
            file_path: Chemin vers le fichier CSV
            **kwargs: Arguments supplémentaires pour pd.read_csv()
            
        Returns:
            pd.DataFrame: Données chargées
        """
        try:
            # Si c'est un objet fichier téléversé
            if hasattr(file_path, 'read'):
                return pd.read_csv(file_path, **kwargs)
            # Si c'est un chemin de fichier
            else:
                return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du CSV {file_path}: {e}")
            raise
    
    def load_excel(self, file_path: Union[str, Path], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Charge un fichier Excel dans un dictionnaire de DataFrames pandas.
        
        Args:
            file_path: Chemin vers le fichier Excel
            **kwargs: Arguments supplémentaires pour pd.ExcelFile()
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionnaire des feuilles du classeur
        """
        try:
            # Si c'est un objet fichier téléversé
            if hasattr(file_path, 'read'):
                excel_file = pd.ExcelFile(file_path, **kwargs)
            # Si c'est un chemin de fichier
            else:
                excel_file = pd.ExcelFile(file_path, **kwargs)
                
            # Charger toutes les feuilles
            return {sheet_name: excel_file.parse(sheet_name) 
                   for sheet_name in excel_file.sheet_names}
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier Excel {file_path}: {e}")
            raise
    
    def process_uploaded_files(self, uploaded_files: Dict[str, any]) -> Dict[str, pd.DataFrame]:
        """
        Traite les fichiers téléversés et charge les données.
        
        Args:
            uploaded_files: Dictionnaire des fichiers téléversés (clé: type, valeur: fichier)
            
        Returns:
            Dict[str, pd.DataFrame]: Données chargées avec les clés correspondantes
        """
        loaded_data = {}
        
        for file_type, uploaded_file in uploaded_files.items():
            if uploaded_file is not None:
                try:
                    # Enregistrer le fichier téléversé
                    file_path = self.save_uploaded_file(uploaded_file, file_type)
                    
                    # Charger les données en fonction du type de fichier
                    if file_path.suffix.lower() == '.csv':
                        df = self.load_csv(file_path, encoding='latin1')
                        loaded_data[file_type] = df
                        
                        # Enregistrer les métadonnées
                        self.metadata[file_type] = {
                            'rows': len(df),
                            'columns': list(df.columns),
                            'missing_values': df.isnull().sum().to_dict(),
                            'dtypes': df.dtypes.astype(str).to_dict()
                        }
                        
                    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                        dfs = self.load_excel(file_path)
                        for sheet_name, df in dfs.items():
                            key = f"{file_type}_{sheet_name}"
                            loaded_data[key] = df
                            
                            # Enregistrer les métadonnées
                            self.metadata[key] = {
                                'rows': len(df),
                                'columns': list(df.columns),
                                'missing_values': df.isnull().sum().to_dict(),
                                'dtypes': df.dtypes.astype(str).to_dict()
                            }
                            
                    logger.info(f"Fichier {file_type} chargé avec succès")
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du fichier {file_type}: {e}")
                    st.error(f"Erreur lors du traitement du fichier {file_type}: {e}")
        
        return loaded_data
    
    def get_data_summary(self) -> Dict[str, dict]:
        """
        Retourne un résumé des données chargées.
        
        Returns:
            Dict[str, dict]: Résumé des données par type de fichier
        """
        summary = {}
        
        for key, df in self.data.items():
            summary[key] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else {}
            }
            
        return summary
    
    def validate_data_requirements(self, required_files: List[str]) -> Tuple[bool, List[str]]:
        """
        Vérifie si tous les fichiers requis sont chargés.
        
        Args:
            required_files: Liste des types de fichiers requis
            
        Returns:
            Tuple[bool, List[str]]: (Tous les fichiers sont présents, Liste des fichiers manquants)
        """
        missing_files = [f for f in required_files if f not in self.data]
        return (len(missing_files) == 0, missing_files)

# Fonction utilitaire pour afficher un aperçu des données
def display_data_preview(df: pd.DataFrame, title: str = "Aperçu des données") -> None:
    """
    Affiche un aperçu des données dans l'interface Streamlit.
    
    Args:
        df: DataFrame à afficher
        title: Titre de la section
    """
    if df is not None:
        st.subheader(title)
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Nombre de lignes", df.shape[0])
            
        with col2:
            st.metric("Nombre de colonnes", df.shape[1])
