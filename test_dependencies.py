#!/usr/bin/env python3
"""
Test complet pour vérifier que toutes les dépendances sont installées correctement.
"""

def test_imports():
    """Test toutes les importations critiques."""
    print("🧪 Test des importations...")
    
    try:
        import streamlit as st
        print("✅ Streamlit OK")
    except ImportError as e:
        print(f"❌ Streamlit ÉCHEC: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas OK")
    except ImportError as e:
        print(f"❌ Pandas ÉCHEC: {e}")
        return False
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib OK (version {matplotlib.__version__})")
    except ImportError as e:
        print(f"❌ Matplotlib ÉCHEC: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ Numpy OK")
    except ImportError as e:
        print(f"❌ Numpy ÉCHEC: {e}")
        return False
    
    try:
        import plotly
        print("✅ Plotly OK")
    except ImportError as e:
        print(f"❌ Plotly ÉCHEC: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ Seaborn OK")
    except ImportError as e:
        print(f"❌ Seaborn ÉCHEC: {e}")
        return False
    
    try:
        import chardet
        print("✅ Chardet OK")
    except ImportError as e:
        print(f"❌ Chardet ÉCHEC: {e}")
        return False
    
    return True

def test_background_gradient():
    """Test spécifique pour background_gradient."""
    print("\n🎨 Test de background_gradient...")
    
    try:
        import pandas as pd
        
        # Créer un DataFrame test
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })
        
        # Tester background_gradient
        styled = df.style.background_gradient(cmap='YlGnBu', axis=0)
        print("✅ background_gradient avec YlGnBu OK")
        
        # Tester avec différents cmaps
        styled2 = df.style.background_gradient(cmap='viridis', axis=1)
        print("✅ background_gradient avec viridis OK")
        
        # Tester la méthode _compute pour s'assurer que le style peut être calculé
        styled._compute()
        print("✅ Style compute OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        return False
    except Exception as e:
        print(f"❌ Autre erreur: {e}")
        return False

def test_streamlit_dataframe_styling():
    """Test la compatibilité avec streamlit."""
    print("\n📊 Test de compatibilité Streamlit...")
    
    try:
        import pandas as pd
        import streamlit as st
        
        # Simuler un DataFrame comme dans l'application
        df = pd.DataFrame({
            'segment': ['Champions', 'Loyaux', 'Nouveaux'],
            'count': [100, 200, 150],
            'avg_recency': [10, 30, 5],
            'avg_frequency': [15, 8, 2],
            'avg_monetary': [500, 300, 100]
        })
        
        # Tester le style exact utilisé dans l'application
        styled_df = df.style.background_gradient(cmap='YlGnBu', axis=0)
        
        print("✅ Style DataFrame compatible Streamlit OK")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de compatibilité Streamlit: {e}")
        return False

def main():
    print("🚀 Tests de vérification des dépendances et fonctionnalités\n")
    
    # Test des importations
    imports_ok = test_imports()
    
    # Test background_gradient
    bg_gradient_ok = test_background_gradient()
    
    # Test compatibilité Streamlit
    streamlit_ok = test_streamlit_dataframe_styling()
    
    # Résumé
    print("\n" + "="*50)
    print("📋 RÉSUMÉ DES TESTS")
    print("="*50)
    print(f"Importations: {'✅ OK' if imports_ok else '❌ ÉCHEC'}")
    print(f"Background gradient: {'✅ OK' if bg_gradient_ok else '❌ ÉCHEC'}")
    print(f"Compatibilité Streamlit: {'✅ OK' if streamlit_ok else '❌ ÉCHEC'}")
    
    if imports_ok and bg_gradient_ok and streamlit_ok:
        print("\n🎉 Tous les tests sont RÉUSSIS ! L'application devrait fonctionner correctement.")
        return True
    else:
        print("\n⚠️ Certains tests ont ÉCHOUÉ. Vérifiez les erreurs ci-dessus.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
