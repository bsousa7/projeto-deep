"""Avaliação comparativa dos modelos: métricas, matrizes de confusão e mapa geográfico."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"
FIGURAS = RESULTADOS / "figuras"


def calcular_metricas(
    y_true: pd.Series,
    y_pred: np.ndarray,
    nome_modelo: str = "modelo",
) -> dict:
    """Calcula métricas de classificação com foco em F1-macro.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        nome_modelo: Nome para identificação no relatório.

    Returns:
        Dicionário com f1_macro, precisao_macro, revocacao_macro, acuracia.
    """
    metricas = {
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precisao_macro": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "revocacao_macro": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "acuracia": round(accuracy_score(y_true, y_pred), 4),
    }
    print(f"\n{'='*60}")
    print(f"Modelo: {nome_modelo}")
    for chave, valor in metricas.items():
        print(f"  {chave}: {valor:.4f}")
    print("\nRelatório por classe:")
    print(classification_report(y_true, y_pred, zero_division=0))
    return metricas


def plotar_matriz_confusao(
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes: list[str],
    nome_arquivo: str = "matriz_confusao.png",
    titulo: str = "Matriz de Confusão",
) -> Path:
    """Gera e salva a matriz de confusão.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        classes: Lista de nomes das classes na ordem dos rótulos.
        nome_arquivo: Nome do arquivo de saída (salvo em resultados/figuras/).
        titulo: Título do gráfico.

    Returns:
        Caminho do arquivo salvo.
    """
    FIGURAS.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(titulo)
    plt.tight_layout()

    caminho = FIGURAS / nome_arquivo
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusão salva em {caminho}")
    return caminho


def comparar_modelos(
    y_test: pd.Series,
    pred_baseline: np.ndarray,
    pred_transformer: np.ndarray,
    classes: list[str],
) -> dict:
    """Gera tabela comparativa e matrizes de confusão dos dois modelos.

    Args:
        y_test: Rótulos verdadeiros do conjunto de teste.
        pred_baseline: Predições do baseline (TF-IDF + linear).
        pred_transformer: Predições do Transformer (BERTimbau).
        classes: Lista de classes.

    Returns:
        Dicionário com métricas de baseline, transformer e ganho absoluto.
    """
    met_base = calcular_metricas(y_test, pred_baseline, "Baseline (TF-IDF + LogReg)")
    met_transformer = calcular_metricas(y_test, pred_transformer, "Transformer (BERTimbau)")

    ganho = round(met_transformer["f1_macro"] - met_base["f1_macro"], 4)
    print(f"\nGanho F1-macro (Transformer − Baseline): {ganho:+.4f}")

    plotar_matriz_confusao(
        y_test, pred_baseline, classes,
        nome_arquivo="matriz_confusao_baseline.png",
        titulo="Matriz de Confusão — Baseline",
    )
    plotar_matriz_confusao(
        y_test, pred_transformer, classes,
        nome_arquivo="matriz_confusao_transformer.png",
        titulo="Matriz de Confusão — Transformer",
    )

    resultado = {
        "baseline": met_base,
        "transformer": met_transformer,
        "ganho_f1_macro": ganho,
    }

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(RESULTADOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"Métricas consolidadas salvas em {RESULTADOS / 'metricas.json'}")

    return resultado


def plotar_f1_por_classe(
    y_test: pd.Series,
    pred_baseline: np.ndarray,
    pred_transformer: np.ndarray,
    classes: list[str],
) -> Path:
    """Gráfico de barras comparando F1 por classe entre baseline e Transformer.

    Returns:
        Caminho do arquivo salvo.
    """
    FIGURAS.mkdir(parents=True, exist_ok=True)

    f1_base = f1_score(y_test, pred_baseline, labels=classes, average=None, zero_division=0)
    f1_trans = f1_score(y_test, pred_transformer, labels=classes, average=None, zero_division=0)

    x = np.arange(len(classes))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 1.2), 5))
    ax.bar(x - largura / 2, f1_base, largura, label="Baseline")
    ax.bar(x + largura / 2, f1_trans, largura, label="Transformer")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("F1 por classe")
    ax.set_title("Comparação de F1 por Classe")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    caminho = FIGURAS / "f1_por_classe.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Gráfico F1 por classe salvo em {caminho}")
    return caminho


def gerar_mapa_calor(
    df: pd.DataFrame,
    col_categoria: str = "categoria_prevista",
    col_lat: str = "latitude",
    col_lon: str = "longitude",
    nome_arquivo: str = "mapa_demandas.html",
) -> Path:
    """Gera mapa de calor de demandas por categoria e localização geográfica.

    Requer colunas de latitude e longitude no DataFrame.

    Args:
        df: DataFrame com predições e coordenadas.
        col_categoria: Coluna com a categoria predita.
        col_lat: Coluna de latitude.
        col_lon: Coluna de longitude.
        nome_arquivo: Nome do arquivo HTML de saída.

    Returns:
        Caminho do arquivo salvo.
    """
    import folium
    from folium.plugins import HeatMap

    FIGURAS.mkdir(parents=True, exist_ok=True)
    df_geo = df[[col_lat, col_lon, col_categoria]].dropna()

    if df_geo.empty:
        raise ValueError("DataFrame sem coordenadas válidas para o mapa.")

    centro = [df_geo[col_lat].mean(), df_geo[col_lon].mean()]
    mapa = folium.Map(location=centro, zoom_start=12)

    categorias = df_geo[col_categoria].unique()
    cores = plt.cm.get_cmap("tab10", len(categorias))

    for i, categoria in enumerate(categorias):
        subset = df_geo[df_geo[col_categoria] == categoria]
        pontos = subset[[col_lat, col_lon]].values.tolist()
        cor = "#{:02x}{:02x}{:02x}".format(
            *[int(c * 255) for c in cores(i)[:3]]
        )
        feature_group = folium.FeatureGroup(name=categoria)
        HeatMap(pontos, radius=15, blur=10).add_to(feature_group)
        feature_group.add_to(mapa)

    folium.LayerControl().add_to(mapa)

    caminho = FIGURAS / nome_arquivo
    mapa.save(str(caminho))
    print(f"Mapa salvo em {caminho}")
    return caminho
