"""Pré-processamento de texto: limpeza leve (BERT) e agressiva (TF-IDF)."""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# spaCy carregado sob demanda para evitar import pesado em todo contexto
_nlp = None


def _carregar_spacy():
    """Carrega o modelo spaCy pt_core_news_lg (uma vez)."""
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("pt_core_news_lg")
    return _nlp


def _normalizar_unicode(texto: str) -> str:
    """Converte caracteres compostos para forma NFC e remove controles invisíveis."""
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)
    return texto


def _remover_urls(texto: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", texto)


def _remover_numeros(texto: str) -> str:
    return re.sub(r"\b\d+\b", " ", texto)


def limpar_para_bert(texto: str) -> str:
    """Limpeza leve para entrada em modelos Transformer.

    Preserva estrutura de frase, capitalização e pontuação relevante.
    Remove apenas ruídos óbvios (URLs, caracteres de controle).
    """
    if not isinstance(texto, str):
        return ""
    texto = _normalizar_unicode(texto)
    texto = _remover_urls(texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def limpar_para_tfidf(texto: str) -> str:
    """Limpeza agressiva para modelos clássicos (TF-IDF).

    Aplica lowercase, remoção de stopwords e lematização via spaCy.
    """
    if not isinstance(texto, str):
        return ""
    texto = _normalizar_unicode(texto)
    texto = _remover_urls(texto)
    texto = texto.lower()
    texto = _remover_numeros(texto)
    texto = re.sub(r"[^\w\s]", " ", texto)  # remove pontuação

    nlp = _carregar_spacy()
    doc = nlp(texto)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.lemma_) > 2
    ]
    return " ".join(tokens)


def limpar_coluna(
    df: pd.DataFrame,
    coluna: str = "texto",
    modo: str = "bert",
) -> pd.DataFrame:
    """Aplica limpeza a uma coluna do DataFrame.

    Args:
        df: DataFrame de entrada.
        coluna: Nome da coluna de texto.
        modo: 'bert' para limpeza leve ou 'tfidf' para limpeza agressiva.

    Returns:
        DataFrame com a coluna limpa (nova coluna 'texto_limpo').
    """
    fn = limpar_para_bert if modo == "bert" else limpar_para_tfidf
    df = df.copy()
    df["texto_limpo"] = df[coluna].apply(fn)
    return df


def dividir_dados(
    df: pd.DataFrame,
    coluna_texto: str = "texto",
    coluna_label: str = "categoria",
    proporcao_teste: float = 0.2,
    proporcao_val: float = 0.5,
    seed: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split estratificado treino / validação / teste.

    Proporção padrão: 80% treino, 10% validação, 10% teste.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[coluna_texto]
    y = df[coluna_label]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=proporcao_teste, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=proporcao_val, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def calcular_kappa(rotulos_a: list, rotulos_b: list) -> float:
    """Calcula o kappa de Cohen entre dois conjuntos de rótulos.

    Args:
        rotulos_a: Rótulos do anotador A.
        rotulos_b: Rótulos do anotador B.

    Returns:
        Kappa de Cohen (float entre -1 e 1).
    """
    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(rotulos_a, rotulos_b)
    return kappa
