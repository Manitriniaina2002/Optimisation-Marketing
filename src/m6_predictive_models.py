import logging, sys, os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('m6_predictive.log')])
logger = logging.getLogger(__name__)


def build_churn_dataset(customers, sales, horizon_days=90):
    # TODO: labellisation churn selon logique métier
    agg = sales.groupby('customer_id').agg(total_spent=('total_amount','sum'), orders=('order_id','nunique')).reset_index()
    df = customers.merge(agg, on='customer_id', how='left').fillna(0)
    df['churn'] = (df['orders'] == 0).astype(int)
    return df


def train_churn(df):
    X = df[['age','total_spent','orders']].fillna(0)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    logger.info(f'AUC: {auc:.3f}')
    return clf


def render_sales_objective_prediction_section():
    """Render the Streamlit UI section for commercial objective prediction (M6)."""
    st.header(" Prédiction d'objectifs commerciaux")
    try:
        import joblib
        import pandas as pd

        # Zone d'import et détection du modèle
        st.subheader("Chargement du Modèle")
        st.caption("Le fichier attendu est 'modele_objectif_commercial.pkl'. Vous pouvez l'uploader ou saisir un chemin personnalisé.")

        # Uploader de modèle
        uploaded_model = st.file_uploader("Uploader le modèle (.pkl)", type=["pkl"], key="m6_model_uploader")
        model_path = None
        models_dir = os.path.join("models")
        os.makedirs(models_dir, exist_ok=True)

        if uploaded_model is not None:
            save_path = os.path.join(models_dir, "modele_objectif_commercial.pkl")
            with open(save_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.success(f"Modèle uploadé et sauvegardé: {save_path}")
            model_path = save_path

        # Champ de chemin manuel
        with st.expander("Options avancées: Chemin personnalisé du modèle"):
            manual_path = st.text_input("Chemin du modèle (.pkl)", value="")
            if manual_path:
                model_path = manual_path

        # Recherche dans les chemins par défaut si aucun chemin fourni
        if not model_path:
            candidates = [
                "modele_objectif_commercial.pkl",
                os.path.join("models", "modele_objectif_commercial.pkl"),
                os.path.join("output", "modele_objectif_commercial.pkl"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    model_path = p
                    break

        if not model_path or not os.path.exists(model_path):
            st.warning("Aucun fichier modèle trouvé. Uploadez un fichier .pkl ou saisissez un chemin valide ci-dessus.")
            return

        # Charger le modèle
        model = joblib.load(model_path)
        st.info(f"Modèle chargé depuis: {model_path}")

        # Créer un formulaire pour la prédiction
        with st.form("prediction_form"):
            st.subheader("Entrez les données pour la prédiction")

            col1, col2 = st.columns(2)

            with col1:
                ca_mensuel = st.number_input("CA mensuel (Ar)", min_value=0, value=880000)
                nb_ventes = st.number_input("Nombre de ventes", min_value=1, value=29)
                panier_moyen = st.number_input("Panier moyen (Ar)", min_value=0, value=30300)

            with col2:
                taux_conversion = st.number_input("Taux de conversion (%)", min_value=0.0, value=27.0, step=0.1)
                prospects_qualifies = st.number_input("Nombre de prospects qualifiés", min_value=1, value=170)
                taux_transformation = st.number_input("Taux de transformation (%)", min_value=0.0, value=14.0, step=0.1)

            submitted = st.form_submit_button("Prédire l'atteinte des objectifs")

            if submitted:
                # Préparer les données pour la prédiction
                new_data = pd.DataFrame({
                    "CA_mensuel": [ca_mensuel],
                    "nb_ventes": [nb_ventes],
                    "panier_moyen": [panier_moyen],
                    "taux_conversion": [taux_conversion],
                    "prospects_qualifies": [prospects_qualifies],
                    "taux_transformation": [taux_transformation]
                })

                # Faire la prédiction
                prediction = model.predict(new_data)
                proba = model.predict_proba(new_data)

                # Afficher les résultats
                st.subheader("Résultats de la prédiction")

                if prediction[0] == 1:
                    st.success(" Objectif commercial atteint !")
                else:
                    st.error(" Objectif commercial non atteint")

                # Afficher les probabilités
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilité d'échec", f"{proba[0][0]*100:.1f}%")
                with col2:
                    st.metric("Probabilité de succès", f"{proba[0][1]*100:.1f}%")

                # Afficher les recommandations
                st.subheader("Recommandations")
                if prediction[0] == 1:
                    st.markdown("""
                    - Continuez vos efforts de vente actuels
                    - Envisagez d'augmenter vos objectifs
                    - Récompensez votre équipe pour ces bons résultats
                    """)
                else:
                    st.markdown("""
                    - Analysez les points faibles de votre stratégie actuelle
                    - Envisagez des promotions ciblées
                    - Améliorez votre taux de conversion
                    - Augmentez votre nombre de prospects qualifiés
                    """)

                # Afficher les données utilisées pour la prédiction
                with st.expander("Voir les données utilisées pour la prédiction"):
                    st.write("Données d'entrée :")
                    st.json({
                        "CA mensuel (Ar)": f"{ca_mensuel:,}",
                        "Nombre de ventes": nb_ventes,
                        "Panier moyen (Ar)": f"{panier_moyen:,}",
                        "Taux de conversion (%)": taux_conversion,
                        "Prospects qualifiés": prospects_qualifies,
                        "Taux de transformation (%)": taux_transformation
                    })

    except Exception as e:
        st.warning(" Impossible de charger le modèle de prédiction. Assurez-vous que le fichier 'modele_objectif_commercial.pkl' est présent dans le répertoire.")
        st.error(f"Erreur: {str(e)}")


def main():
    os.makedirs('output', exist_ok=True)
    customers = pd.read_csv('output/customers_clean.csv')
    sales = pd.read_csv('output/sales_clean.csv')
    df = build_churn_dataset(customers, sales)
    _ = train_churn(df)
    # Note: The Streamlit UI is exposed via
    # `render_sales_objective_prediction_section()` to be called from the dashboard (M8).
    # We do not call it here to avoid Streamlit errors when running this module as a script.

if __name__ == '__main__':
    main()
