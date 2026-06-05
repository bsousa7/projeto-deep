"""Baseline clássico: TF-IDF + Regressão Logística ou SVM."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"


def construir_pipeline(
    modelo: Literal["logistic", "svm"] = "logistic",
    seed: int = RANDOM_STATE,
) -> Pipeline:
    """Cria pipeline TF-IDF + classificador.

    Args:
        modelo: 'logistic' para Regressão Logística ou 'svm' para LinearSVC.
        seed: Semente aleatória para reprodutibilidade.

    Returns:
        Pipeline sklearn pronto para fit/predict.
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )

    if modelo == "logistic":
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=seed,
            n_jobs=None,
        )
    elif modelo == "svm":
        clf = LinearSVC(max_iter=2000, C=1.0, random_state=seed)
    else:
        raise ValueError(f"Modelo desconhecido: '{modelo}'. Use 'logistic' ou 'svm'.")

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def treinar_baseline(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    modelo: Literal["logistic", "svm"] = "logistic",
    seed: int = RANDOM_STATE,
) -> tuple[Pipeline, np.ndarray]:
    """Treina o baseline e retorna o pipeline treinado e as predições no teste.

    Args:
        X_train: Textos de treino (já limpos para TF-IDF).
        y_train: Rótulos de treino.
        X_test: Textos de teste.
        modelo: 'logistic' ou 'svm'.
        seed: Semente aleatória.

    Returns:
        (pipeline_treinado, predicoes_teste)
    """
    pipe = construir_pipeline(modelo=modelo, seed=seed)
    pipe.fit(X_train, y_train)
    predicoes = pipe.predict(X_test)
    return pipe, predicoes


def avaliar(
    y_test: pd.Series,
    predicoes: np.ndarray,
    nome_modelo: str = "baseline",
) -> dict:
    """Calcula e exibe métricas do baseline.

    Args:
        y_test: Rótulos verdadeiros.
        predicoes: Rótulos preditos.
        nome_modelo: Identificador para o log.

    Returns:
        Dicionário com f1_macro, precisao_macro, revocacao_macro, acuracia.
    """
    f1 = f1_score(y_test, predicoes, average="macro", zero_division=0)
    relatorio = classification_report(y_test, predicoes, zero_division=0)
    print(f"\n{'='*60}")
    print(f"Modelo: {nome_modelo}")
    print(f"F1-macro: {f1:.4f}")
    print(relatorio)

    metricas = {
        "f1_macro": round(f1, 4),
        "precisao_macro": round(
            precision_score(y_test, predicoes, average="macro", zero_division=0), 4
        ),
        "revocacao_macro": round(
            recall_score(y_test, predicoes, average="macro", zero_division=0), 4
        ),
        "acuracia": round(accuracy_score(y_test, predicoes), 4),
    }
    return metricas


if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.preprocessamento.limpeza import dividir_dados, limpar_para_tfidf

    DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "interim"
    arquivo = DATA_PROCESSED / "acordaos_filtrados.parquet"

    if not arquivo.exists():
        print(f"Arquivo não encontrado: {arquivo}")
        print("Execute primeiro: python src/aquisicao/baixar_csvs.py && python src/preprocessamento/filtrar_tematico.py")
        sys.exit(1)

    df = pd.read_parquet(arquivo, columns=["sumario", "label"])
    df["texto_tfidf"] = df["sumario"].apply(limpar_para_tfidf)

    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(
        df, coluna_texto="texto_tfidf", coluna_label="label"
    )

    pipe, predicoes = treinar_baseline(X_train, y_train, X_test)
    metricas = avaliar(y_test, predicoes, nome_modelo="TF-IDF + LogisticRegression")

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    saida_json = RESULTADOS / "metricas_baseline.json"
    with open(saida_json, "w", encoding="utf-8") as f:
        json.dump({"baseline": metricas}, f, ensure_ascii=False, indent=2)
    print(f"\nMétricas salvas em {saida_json}")
