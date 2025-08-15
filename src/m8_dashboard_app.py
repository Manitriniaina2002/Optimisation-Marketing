import os
import sys
import glob
import logging
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_module(module_name: str, func: str = 'main'):
    try:
        # Support running from project root or src/
        if module_name.startswith('src.'):
            mod = __import__(module_name, fromlist=[func])
        else:
            sys.path.append(os.path.dirname(__file__))
            mod = __import__(module_name, fromlist=[func])
        getattr(mod, func)()
        return True, None
    except Exception as e:
        logger.exception(f"Failed running {module_name}.{func}")
        return False, str(e)


def load_csv_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # Return empty DataFrame to signal an empty file without crashing the app
        return pd.DataFrame()
    except Exception as e:
        # Re-raise other exceptions to surface real parsing problems
        raise


def file_list(pattern):
    return sorted(glob.glob(pattern))


def section_data_preview():
    st.subheader('Data – Fichiers bruts (data/)')
    files = file_list(os.path.join('data', '*.csv'))
    if not files:
        st.info("Aucun CSV trouvé dans data/.")
        return
    for f in files:
        with st.expander(os.path.basename(f)):
            try:
                df = pd.read_csv(f)
                st.dataframe(df.head(50))
            except Exception as e:
                st.error(f"Lecture impossible: {e}")


def section_clean_outputs():
    st.subheader('M2 – Fichiers nettoyés (output/*_clean.csv)')
    for name in ['customers_clean.csv', 'products_clean.csv', 'sales_clean.csv', 'marketing_clean.csv']:
        path = os.path.join('output', name)
        with st.expander(name):
            df = load_csv_if_exists(path)
            if df is None:
                st.warning('Non disponible. Exécute M2.')
            else:
                if df.empty:
                    st.warning('Fichier vide (aucune colonne/ligne). Vérifie le contenu source dans data/ puis relance M2.')
                else:
                    st.dataframe(df.head(100))


def section_segmentation():
    st.subheader('M3 – Segmentation')
    seg_path = os.path.join('output', 'customer_segments.csv')
    df = load_csv_if_exists(seg_path)
    if df is None:
        st.warning('customer_segments.csv non disponible. Exécute M3.')
    else:
        if df.empty:
            st.warning('customer_segments.csv est vide. Assure-toi que M2 a généré des fichiers non vides puis relance M3.')
        else:
            st.dataframe(df.head(200))


def section_personas():
    st.subheader('M4 – Personas & Visualisations')
    img_names = ['segment_distribution.png', 'avg_spend_by_segment.png']
    cols = st.columns(2)
    for i, name in enumerate(img_names):
        p = os.path.join('output', name)
        with cols[i % 2]:
            if os.path.exists(p):
                st.image(p, caption=name, use_container_width=True)
            else:
                st.info(f"{name} non généré.")
    report_path = os.path.join('output', 'customer_personas_report.txt')
    with st.expander('customer_personas_report.txt'):
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                st.text(f.read())
        else:
            st.info('Rapport non disponible. Exécute M4.')


def section_marketing():
    st.subheader('M5 – KPIs Marketing & Benchmarks M1')
    for name in ['campaign_kpis.csv', 'kpi_benchmark.csv']:
        path = os.path.join('output', name)
        with st.expander(name):
            df = load_csv_if_exists(path)
            if df is None:
                st.warning('Non disponible. Exécute M5.')
            else:
                if df.empty:
                    st.warning(f'{name} est vide. Vérifie marketing_clean.csv puis relance M5.')
                else:
                    st.dataframe(df)
    txt = os.path.join('output', 'kpi_summary.txt')
    with st.expander('kpi_summary.txt'):
        if os.path.exists(txt):
            with open(txt, 'r', encoding='utf-8') as f:
                st.text(f.read())
        else:
            st.info('Résumé non disponible.')


def main():
    st.set_page_config(page_title='TeeTech Analytics', layout='wide')
    st.title('TeeTech – Marketing Analytics')
    st.caption('Pipeline M2→M5, visualisations et rapports')

    with st.sidebar:
        st.header('Pipeline')
        if st.button('Exécuter M2 – Nettoyage (data → output/*_clean.csv)'):
            ok, err = run_module('src.m2_data_preparation')
            st.success('M2 terminé') if ok else st.error(f"M2 erreur: {err}")

        if st.button('Exécuter M3 – Segmentation'):
            ok, err = run_module('src.m3_customer_segmentation')
            st.success('M3 terminé') if ok else st.error(f"M3 erreur: {err}")

        if st.button('Exécuter M4 – Profiling & Personas'):
            ok, err = run_module('src.m4_customer_profiling')
            st.success('M4 terminé') if ok else st.error(f"M4 erreur: {err}")

        if st.button('Exécuter M5 – KPIs Marketing'):
            ok, err = run_module('src.m5_marketing_campaigns')
            st.success('M5 terminé') if ok else st.error(f"M5 erreur: {err}")

        st.markdown('---')
        st.caption('Astuce: dépose tes CSV dans data/ puis lance M2')

    tabs = st.tabs([
        'Data (brut)', 'M2 Outputs', 'M3 Segments', 'M4 Personas', 'M5 KPIs'
    ])
    with tabs[0]:
        section_data_preview()
    with tabs[1]:
        section_clean_outputs()
    with tabs[2]:
        section_segmentation()
    with tabs[3]:
        section_personas()
    with tabs[4]:
        section_marketing()


if __name__ == '__main__':
    main()
