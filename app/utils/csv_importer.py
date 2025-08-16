"""
Module utilitaire pour l'importation robuste des fichiers CSV.
Gère automatiquement la détection du séparateur, l'encodage et le typage des colonnes.
"""

import pandas as pd
import chardet
from io import StringIO
import streamlit as st

def detect_separator(content):
    """Détecte le séparateur utilisé dans le fichier CSV."""
    # Essaye de détecter le séparateur en analysant la première ligne
    first_line = content.split('\n')[0]
    
    # Compte les occurrences de chaque séparateur potentiel
    separators = {',': first_line.count(','), ';': first_line.count(';'), '\t': first_line.count('\t')}
    
    # Retourne le séparateur le plus fréquent
    return max(separators.items(), key=lambda x: x[1])[0]

def detect_encoding(file_content):
    """Détecte l'encodage du fichier."""
    result = chardet.detect(file_content)
    return result['encoding']

def clean_column_name(column_name):
    """Nettoie le nom des colonnes en minuscules et supprime les espaces."""
    return str(column_name).strip().lower().replace(' ', '_')

def map_column_names(df, expected_columns):
    """Tente de mapper les noms de colonnes réels aux noms attendus."""
    # Nettoyer les noms de colonnes
    df_columns = [clean_column_name(col) for col in df.columns]
    expected_columns = [clean_column_name(col) for col in expected_columns]
    
    # Créer un mapping des colonnes trouvées vers les colonnes attendues
    column_mapping = {}
    
    # Dictionnaire des alias courants pour les colonnes importantes
    column_aliases = {
        'order_id': ['id_commande', 'commande_id', 'num_commande'],
        'customer_id': ['client_id', 'id_client', 'customer', 'client'],
        'order_date': ['date_commande', 'date', 'dateorder', 'orderdate'],
        'amount': ['montant', 'total', 'prix_total', 'total_ttc'],
        'price': ['prix', 'prix_unitaire', 'unit_price'],
        'quantity': ['quantite', 'qte', 'qty']
    }
    
    for expected in expected_columns:
        # Essayer de trouver une correspondance exacte
        if expected in df_columns:
            column_mapping[expected] = expected
            continue
            
        # Vérifier les alias connus
        if expected in column_aliases:
            for alias in column_aliases[expected]:
                if alias in df_columns:
                    column_mapping[expected] = alias
                    break
        
        # Si toujours pas trouvé, essayer une correspondance partielle
        if expected not in column_mapping:
            for col in df_columns:
                if expected in col or col in expected:
                    column_mapping[expected] = col
                    break
    
    # Renommer les colonnes si nécessaire
    if column_mapping:
        df = df.rename(columns={v: k for k, v in column_mapping.items() if k != v})
    
    return df

def convert_date_columns(df, date_columns):
    """Convertit les colonnes de date au bon format."""
    date_converters = [
        lambda x: pd.to_datetime(x, dayfirst=True, errors='coerce'),  # Format JJ/MM/AAAA
        lambda x: pd.to_datetime(x, yearfirst=True, errors='coerce'),  # Format AAAA-MM-JJ
        lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'),
        lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'),
        lambda x: pd.to_datetime(x, format='%d-%m-%Y', errors='coerce'),
        lambda x: pd.to_datetime(x, format='%Y/%m/%d', errors='coerce')
    ]
    
    for col in date_columns:
        if col in df.columns:
            # Essayer différents formats de date
            for converter in date_converters:
                converted = converter(df[col].astype(str))
                if not converted.isna().all():  # Si au moins une conversion a réussi
                    df[col] = converted
                    break
            
            # Vérifier si la conversion a réussi
            if df[col].isna().all():
                st.warning(f"⚠️ La colonne '{col}' n'a pas pu être convertie en date. Vérifiez le format.")
            
    return df

def convert_numeric_columns(df, numeric_columns):
    """Convertit les colonnes numériques au bon format."""
    for col in numeric_columns:
        if col in df.columns:
            # Sauvegarder le type original pour les messages d'erreur
            original_dtype = str(df[col].dtype)
            
            # Remplacer les virgules par des points pour les nombres décimaux
            # et traiter les chaînes vides comme des valeurs manquantes
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.')
                # Remplacer les chaînes vides et 'nan' par NaN
                df[col] = df[col].replace(['', 'nan', 'None', 'null'], pd.NA)
            
            # Convertir en numérique
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Vérifier les valeurs manquantes après conversion
            na_count = df[col].isna().sum()
            if na_count > 0:
                st.warning(f"⚠️ {na_count} valeurs non numériques détectées dans la colonne '{col}'. Elles ont été remplacées par NaN.")
    
    return df

def load_csv(file_uploader, expected_columns, date_columns=None, numeric_columns=None, required_columns=None):
    """
    Charge un fichier CSV de manière robuste.
    
    Args:
        file_uploader: Le fichier téléchargé via st.file_uploader
        expected_columns: Liste des colonnes attendues
        date_columns: Liste des colonnes de date
        numeric_columns: Liste des colonnes numériques
        required_columns: Colonnes obligatoires (doivent être présentes et non vides)
    
    Returns:
        DataFrame: Les données chargées ou None en cas d'erreur
    """
    if file_uploader is None:
        st.warning("Aucun fichier n'a été téléchargé.")
        return None
    
    # Initialiser les listes vides si None
    date_columns = date_columns or []
    numeric_columns = numeric_columns or []
    required_columns = required_columns or []
    
    # Journalisation
    st.info(f"Chargement du fichier {file_uploader.name}...")
    
    try:
        # Lire le contenu brut du fichier
        file_content = file_uploader.getvalue()
        
        # Détecter l'encodage
        try:
            encoding = detect_encoding(file_content)
            content = file_content.decode(encoding)
        except Exception as e:
            st.error("❌ Impossible de décoder le fichier. L'encodage n'a pas pu être détecté.")
            st.error(f"Détails de l'erreur : {str(e)}")
            return None
        
        # Détecter le séparateur
        try:
            sep = detect_separator(content)
            st.info(f"Séparateur détecté : '{sep}'")
        except Exception as e:
            st.warning("⚠️ Impossible de détecter le séparateur. Utilisation de la virgule par défaut.")
            sep = ','
        
        # Lire le fichier CSV
        try:
            df = pd.read_csv(StringIO(content), sep=sep, dtype=str, encoding=encoding, on_bad_lines='warn')
            st.success(f"✅ Fichier chargé avec succès. {len(df)} lignes trouvées.")
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier CSV : {str(e)}")
            return None
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            st.error("❌ Le fichier est vide ou n'a pas pu être lu correctement.")
            return None
        
        # Nettoyer les noms de colonnes
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Afficher les premières lignes pour débogage
        with st.expander("🔍 Aperçu des données brutes (avant traitement)"):
            st.dataframe(df.head())
        
        # Essayer de mapper les colonnes aux noms attendus
        df = map_column_names(df, expected_columns)
        
        # Afficher le mapping des colonnes
        st.info("📋 Mapping des colonnes :")
        col_mapping = {}
        for col in expected_columns:
            col_mapping[col] = "Trouvée" if col in df.columns else "Manquante"
        st.json(col_mapping)
        
        # Convertir les types de données
        if date_columns:
            st.info("🔄 Conversion des colonnes de date...")
            initial_date_cols = [col for col in date_columns if col in df.columns]
            df = convert_date_columns(df, initial_date_cols)
        
        if numeric_columns:
            st.info("🔢 Conversion des colonnes numériques...")
            initial_num_cols = [col for col in numeric_columns if col in df.columns]
            df = convert_numeric_columns(df, initial_num_cols)
        
        # Vérifier les colonnes requises
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            st.error(f"❌ Colonnes obligatoires manquantes : {', '.join(missing_required)}")
            st.error("Ces colonnes sont nécessaires pour l'analyse. Veuillez vérifier votre fichier.")
            return None
        
        # Vérifier les valeurs manquantes dans les colonnes requises
        if required_columns:
            missing_values = df[required_columns].isnull().sum()
            missing_any = missing_values > 0
            
            if missing_any.any():
                st.warning("⚠️ Valeurs manquantes détectées dans les colonnes requises :")
                missing_info = pd.DataFrame({
                    'Colonne': missing_values.index,
                    'Valeurs manquantes': missing_values.values,
                    '% manquants': (missing_values / len(df) * 100).round(1).astype(str) + '%'
                })
                st.dataframe(missing_info)
                
                # Demander confirmation avant de supprimer les lignes
                if st.checkbox("Supprimer les lignes avec des valeurs manquantes dans les colonnes requises ?"):
                    initial_count = len(df)
                    df = df.dropna(subset=required_columns)
                    removed = initial_count - len(df)
                    if removed > 0:
                        st.warning(f"ℹ️ {removed} lignes ont été supprimées car elles contenaient des valeurs manquantes dans les colonnes requises.")
                        st.warning(f"ℹ️ {len(df)} lignes restantes après nettoyage.")
                else:
                    st.warning("⚠️ Les valeurs manquantes peuvent affecter les analyses ultérieures.")
        
        # Vérifier les colonnes critiques pour les analyses
        critical_columns = {
            'order_date': 'Nécessaire pour les analyses temporelles',
            'amount': 'Nécessaire pour le calcul du chiffre d\'affaires',
            'price': 'Nécessaire si amount n\'est pas disponible',
            'quantity': 'Nécessaire si amount n\'est pas disponible'
        }
        
        missing_critical = []
        for col, desc in critical_columns.items():
            if col not in df.columns and col in expected_columns:
                missing_critical.append(f"- {col} : {desc}")
        
        if missing_critical:
            st.warning("⚠️ Colonnes critiques manquantes pour certaines analyses :")
            st.markdown("\n".join(missing_critical))
        
        # Générer des statistiques de base pour les colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            st.info("📊 Statistiques descriptives des colonnes numériques :")
            st.dataframe(df[numeric_cols].describe().round(2))
        
        # Vérifier les dates si présentes
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols:
            st.info("📅 Plage de dates détectée :")
            for col in date_cols:
                min_date = df[col].min()
                max_date = df[col].max()
                st.write(f"- {col}: de {min_date} à {max_date} ({(max_date - min_date).days} jours)")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Une erreur inattendue s'est produite lors du chargement du fichier : {str(e)}")
        st.error("Veuillez vérifier le format de votre fichier et réessayer.")
        import traceback
        st.text(traceback.format_exc())
        return None
