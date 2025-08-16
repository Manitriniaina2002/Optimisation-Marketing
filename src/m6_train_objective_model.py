import os
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

EXPECTED_FEATURES = [
    "CA_mensuel",
    "nb_ventes",
    "panier_moyen",
    "taux_conversion",
    "prospects_qualifies",
    "taux_transformation",
]
TARGET_NAME = "objectif_atteint"
DEFAULT_MODEL_PATH = Path("models") / "modele_objectif_commercial.pkl"


def _ensure_models_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _generate_synthetic_data(n: int = 500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    # Generate plausible business KPIs
    prospects = rng.integers(50, 1000, size=n)
    taux_conv = rng.normal(10, 4, size=n).clip(0.5, 60)
    nb_ventes = np.maximum(1, (prospects * taux_conv / 100).astype(int))
    panier_moyen = rng.normal(30000, 8000, size=n).clip(5000, 150000)
    ca_mensuel = (nb_ventes * panier_moyen).astype(int)
    taux_transfo = rng.normal(15, 5, size=n).clip(0.5, 80)

    # Latent score combining drivers of success
    score = (
        0.00001 * ca_mensuel
        + 0.02 * nb_ventes
        + 0.00002 * panier_moyen
        + 0.08 * taux_conv
        + 0.0005 * prospects
        + 0.06 * taux_transfo
        + rng.normal(0, 0.5, size=n)
    )
    # Threshold to create binary target
    thresh = np.median(score)
    objectif_atteint = (score > thresh).astype(int)

    df = pd.DataFrame(
        {
            "CA_mensuel": ca_mensuel,
            "nb_ventes": nb_ventes,
            "panier_moyen": panier_moyen,
            "taux_conversion": taux_conv,
            "prospects_qualifies": prospects,
            "taux_transformation": taux_transfo,
            TARGET_NAME: objectif_atteint,
        }
    )
    return df


def _load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable: {data_path}")
    df = pd.read_csv(data_path)
    return df


def _validate_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le dataset: {missing}. Colonnes attendues: {EXPECTED_FEATURES}"
        )
    if TARGET_NAME not in df.columns:
        raise ValueError(
            f"Colonne cible '{TARGET_NAME}' manquante. Ajoutez une colonne binaire 0/1 indiquant l'atteinte de l'objectif."
        )

    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGET_NAME].astype(int)

    for c in EXPECTED_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    return X, y


def train_and_save_model(X: pd.DataFrame, y: pd.Series, out_path: Path = DEFAULT_MODEL_PATH) -> None:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
            ),
        ]
    )

    # Basic split for validation
    stratify = y if y.nunique() >= 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    pipe.fit(X_train, y_train)

    # Simple report
    try:
        y_pred = pipe.predict(X_test)
        report = classification_report(y_test, y_pred)
        print("Evaluation du modèle (jeu de test):\n", report)
    except Exception:
        pass

    _ensure_models_dir(out_path)
    joblib.dump(pipe, out_path)
    print(f"✅ Modèle sauvegardé: {out_path}")


def train_objective_model(data_csv: str | None = None, output_path: str | None = None) -> Path:
    out_path = Path(output_path) if output_path else DEFAULT_MODEL_PATH

    if data_csv:
        df = _load_dataset(Path(data_csv))
        X, y = _validate_and_prepare(df)
    else:
        print("Aucun dataset fourni. Génération de données synthétiques…")
        df = _generate_synthetic_data()
        X, y = _validate_and_prepare(df)

    train_and_save_model(X, y, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Entraîner le modèle d'atteinte d'objectifs commerciaux et sauvegarder models/modele_objectif_commercial.pkl"
        )
    )
    parser.add_argument(
        "--data",
        dest="data",
        type=str,
        default=None,
        help=(
            "Chemin du CSV contenant les colonnes: "
            + ", ".join(EXPECTED_FEATURES)
            + f" et la cible '{TARGET_NAME}'"
        ),
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Chemin de sortie du modèle (.pkl). Par défaut: models/modele_objectif_commercial.pkl",
    )

    args = parser.parse_args()

    path = train_objective_model(args.data, args.output)
    print(f"Modèle entraîné et sauvegardé à: {path}")


if __name__ == "__main__":
    main()
