"""Pré-processamento de texto: limpeza leve (BERT) e agressiva (TF-IDF), head+tail."""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Stopwords jurídicas carregadas sob demanda
_STOPWORDS = None


def _carregar_stopwords() -> set:
    """Carrega stopwords PT-BR do nltk + termos jurídicos irrelevantes."""
    global _STOPWORDS
    if _STOPWORDS is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            base = set(stopwords.words("portuguese"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            base = set(stopwords.words("portuguese"))

        # Termos muito frequentes em acórdãos que não discriminam o desfecho
        juridicos_irrelevantes = {
            "acórdão", "acordao", "processo", "senhor", "senhora", "ministro",
            "relator", "plenário", "câmara", "tcu", "tribunal", "contas",
            "federal", "brasil", "brasília", "requerente", "requerido",
        }
        _STOPWORDS = base | juridicos_irrelevantes
    return _STOPWORDS


def _normalizar_unicode(texto: str) -> str:
    """Converte para NFC e remove caracteres de controle invisíveis."""
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

    Aplica lowercase, remoção de stopwords jurídicas e pontuação.
    """
    if not isinstance(texto, str):
        return ""
    texto = _normalizar_unicode(texto)
    texto = _remover_urls(texto)
    texto = texto.lower()
    texto = _remover_numeros(texto)
    texto = re.sub(r"[^\w\s]", " ", texto)

    stopwords = _carregar_stopwords()
    tokens = [t for t in texto.split() if t not in stopwords and len(t) > 2]
    return " ".join(tokens)


def truncar_head_tail(texto: str, tokenizer, max_head: int = 128, max_tail: int = 384) -> str:
    """Aplica estratégia head+tail para documentos longos.

    Concatena os primeiros max_head tokens e os últimos max_tail tokens,
    totalizando até 512 tokens para o modelo BERT.

    Referência: Sun et al. (2019) — How to Fine-Tune BERT for Text Classification.

    Args:
        texto: Texto limpo a tokenizar.
        tokenizer: Tokenizador HuggingFace (AutoTokenizer).
        max_head: Número de tokens do início do texto.
        max_tail: Número de tokens do final do texto.

    Returns:
        Texto reconstruído a partir dos tokens head+tail.
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""

    tokens = tokenizer.encode(texto, add_special_tokens=False)

    if len(tokens) <= max_head + max_tail:
        return texto  # texto curto o suficiente — sem truncação

    tokens_ht = tokens[:max_head] + tokens[-max_tail:]
    return tokenizer.decode(tokens_ht, skip_special_tokens=True)


def limpar_coluna(
    df: pd.DataFrame,
    coluna: str = "sumario",
    modo: str = "bert",
) -> pd.DataFrame:
    """Aplica limpeza a uma coluna do DataFrame.

    Args:
        df: DataFrame de entrada.
        coluna: Nome da coluna de texto.
        modo: 'bert' para limpeza leve ou 'tfidf' para limpeza agressiva.

    Returns:
        DataFrame com nova coluna 'texto_limpo'.
    """
    fn = limpar_para_bert if modo == "bert" else limpar_para_tfidf
    df = df.copy()
    df["texto_limpo"] = df[coluna].apply(fn)
    return df


def dividir_dados(
    df: pd.DataFrame,
    coluna_texto: str = "sumario",
    coluna_label: str = "label",
    proporcao_val_teste: float = 0.30,
    proporcao_teste_no_temp: float = 0.50,
    seed: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split estratificado 70% treino / 15% validação / 15% teste.

    Args:
        df: DataFrame com texto e label.
        coluna_texto: Coluna de entrada do modelo.
        coluna_label: Coluna de rótulos.
        proporcao_val_teste: Fração do total para val+teste (padrão 0.30).
        proporcao_teste_no_temp: Fração de val+teste destinada ao teste (padrão 0.50 → 15/15).
        seed: Semente para reprodutibilidade.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[coluna_texto]
    y = df[coluna_label]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=proporcao_val_teste, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=proporcao_teste_no_temp, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
