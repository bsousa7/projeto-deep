"""Avaliação comparativa dos modelos: métricas, matrizes de confusão e LIME."""

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


def calcular_metricas_por_classe(
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes: list[str],
    nome_modelo: str = "modelo",
) -> pd.DataFrame:
    """Calcula Precisão, Revocação e F1 por classe individualmente.

    Essencial para corpus desbalanceado: métricas macro ocultam
    o colapso em classes minoritárias (F1=0 em 'Regular').
    """
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, suporte = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    df = pd.DataFrame({
        "Classe":    classes,
        "Suporte":   suporte.astype(int),
        "Precisão":  [round(p, 4) for p in prec],
        "Revocação": [round(r, 4) for r in rec],
        "F1":        [round(f, 4) for f in f1],
    })
    print(f"\n{'='*60}")
    print(f"Métricas por classe — {nome_modelo}")
    print(df.to_string(index=False))
    return df


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
        classes: Lista de nomes das classes.
        nome_arquivo: Nome do arquivo de saída em resultados/figuras/.
        titulo: Título do gráfico.

    Returns:
        Caminho do arquivo salvo.
    """
    FIGURAS.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 2), max(5, len(classes) * 1.5)))
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
        pred_transformer: Predições do Transformer (LegalBert-pt).
        classes: Lista de classes.

    Returns:
        Dicionário com métricas de baseline, transformer e ganho absoluto.
    """
    met_base = calcular_metricas(y_test, pred_baseline, "Baseline (TF-IDF + LogReg)")
    met_transformer = calcular_metricas(y_test, pred_transformer, "LegalBert-pt (head+tail)")

    ganho = round(met_transformer["f1_macro"] - met_base["f1_macro"], 4)
    print(f"\nGanho F1-macro (Transformer − Baseline): {ganho:+.4f}")

    plotar_matriz_confusao(
        y_test, pred_baseline, classes,
        nome_arquivo="matriz_confusao_baseline.png",
        titulo="Matriz de Confusão — Baseline (TF-IDF + LogReg)",
    )
    plotar_matriz_confusao(
        y_test, pred_transformer, classes,
        nome_arquivo="matriz_confusao_transformer.png",
        titulo="Matriz de Confusão — LegalBert-pt (head+tail)",
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
    """Gráfico de barras comparando F1 por classe entre baseline e LegalBert-pt.

    Returns:
        Caminho do arquivo salvo.
    """
    FIGURAS.mkdir(parents=True, exist_ok=True)

    f1_base = f1_score(y_test, pred_baseline, labels=classes, average=None, zero_division=0)
    f1_trans = f1_score(y_test, pred_transformer, labels=classes, average=None, zero_division=0)

    x = np.arange(len(classes))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 2), 5))
    ax.bar(x - largura / 2, f1_base, largura, label="Baseline (TF-IDF)")
    ax.bar(x + largura / 2, f1_trans, largura, label="LegalBert-pt")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("F1 por classe")
    ax.set_title("Comparação de F1 por Classe — Baseline vs. LegalBert-pt")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    caminho = FIGURAS / "f1_por_classe.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Gráfico F1 por classe salvo em {caminho}")
    return caminho


def otimizar_threshold(
    y_true: pd.Series,
    proba_irregular: np.ndarray,
    classe_irregular: str = "Irregular",
    custo_fn: float = 10.0,
    custo_fp: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Encontra o threshold ótimo com base na assimetria de custo FN vs. FP.

    Conforme Tabela 5 do artigo: o custo de um Falso Negativo (irregularidade
    não detectada) é fiscalmente muito superior ao custo de um Falso Positivo
    (auditoria desnecessária). Varrer thresholds abaixo de 0,5 prioriza a
    revocação da classe 'Irregular' sem re-treinamento.

    Args:
        y_true: Rótulos verdadeiros (binário: classe_irregular vs. resto).
        proba_irregular: Probabilidade prevista para a classe 'Irregular'.
        classe_irregular: Nome da classe de alto custo.
        custo_fn: Peso do Falso Negativo na função de custo total.
        custo_fp: Peso do Falso Positivo na função de custo total.
        thresholds: Array de thresholds a testar (default: 0,05 a 0,95).

    Returns:
        Dicionário com threshold_otimo, custo_minimo, revocacao, precisao,
        fn_rate, fp_rate e tabela completa de resultados.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    y_bin = (np.asarray(y_true) == classe_irregular).astype(int)
    resultados = []

    for t in thresholds:
        pred_bin = (proba_irregular >= t).astype(int)
        tp = int(((pred_bin == 1) & (y_bin == 1)).sum())
        fn = int(((pred_bin == 0) & (y_bin == 1)).sum())
        fp = int(((pred_bin == 1) & (y_bin == 0)).sum())
        tn = int(((pred_bin == 0) & (y_bin == 0)).sum())

        revocacao = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisao  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        custo     = custo_fn * fn + custo_fp * fp

        resultados.append({
            "threshold": round(float(t), 2),
            "revocacao": round(revocacao, 4),
            "precisao":  round(precisao, 4),
            "fn": fn, "fp": fp, "tp": tp, "tn": tn,
            "fn_rate": round(fn / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
            "custo_total": round(float(custo), 2),
        })

    df = pd.DataFrame(resultados)
    idx_otimo = df["custo_total"].idxmin()
    otimo = df.loc[idx_otimo].to_dict()

    print(f"\n{'='*60}")
    print(f"Otimização de threshold (custo_fn={custo_fn}×, custo_fp={custo_fp}×)")
    print(f"  Threshold ótimo : {otimo['threshold']:.2f}")
    print(f"  Custo mínimo    : {otimo['custo_total']:.1f}")
    print(f"  Revocação       : {otimo['revocacao']:.4f}  (FN rate: {otimo['fn_rate']:.4f})")
    print(f"  Precisão        : {otimo['precisao']:.4f}")
    print("\nTabela completa:")
    print(df[["threshold", "revocacao", "precisao", "fn_rate", "custo_total"]].to_string(
        index=False
    ))

    return {
        "threshold_otimo": otimo["threshold"],
        "custo_minimo": otimo["custo_total"],
        "revocacao": otimo["revocacao"],
        "precisao": otimo["precisao"],
        "fn_rate": otimo["fn_rate"],
        "fp_rate": round(otimo["fp"] / (otimo["fp"] + otimo["tn"])
                         if (otimo["fp"] + otimo["tn"]) > 0 else 0.0, 4),
        "tabela": df,
    }


def _fn_predict_proba(pipeline):
    """Retorna função de probabilidade compatível com LIME.

    Se o pipeline tem predict_proba (LogReg), usa diretamente.
    Se não (LinearSVC), aplica softmax na decision_function como proxy.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba

    def _softmax(textos):
        scores = pipeline.decision_function(textos)
        scores = np.atleast_2d(scores)
        if scores.shape[1] == 1:
            # Binário: decision_function retorna 1 coluna
            scores = np.hstack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    return _softmax


def explicar_com_lime(
    pipeline_baseline,
    textos_teste: list[str],
    classes: list[str],
    num_features: int = 10,
    num_amostras: int = 5,
    nome_arquivo: str = "lime_explicabilidade.png",
) -> Path:
    """Gera plot LIME mostrando tokens mais preditivos de condenação (Irregular).

    Funciona com LogisticRegression (predict_proba) e LinearSVC (softmax na
    decision_function como proxy de probabilidade).

    Args:
        pipeline_baseline: Pipeline sklearn treinado (TF-IDF + classificador).
        textos_teste: Lista de textos do conjunto de teste.
        classes: Lista de nomes das classes (ordem igual ao LabelEncoder).
        num_features: Número de features (tokens) a destacar por amostra.
        num_amostras: Número de acórdãos a explicar.
        nome_arquivo: Nome do arquivo de saída em resultados/figuras/.

    Returns:
        Caminho do arquivo salvo.
    """
    from lime.lime_text import LimeTextExplainer

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fn_proba = _fn_predict_proba(pipeline_baseline)
    explainer = LimeTextExplainer(class_names=classes)

    fig, axes = plt.subplots(num_amostras, 1, figsize=(12, num_amostras * 3))
    if num_amostras == 1:
        axes = [axes]

    for i, texto in enumerate(textos_teste[:num_amostras]):
        exp = explainer.explain_instance(
            texto,
            fn_proba,
            num_features=num_features,
            labels=[classes.index("Irregular")] if "Irregular" in classes else [0],
        )
        label_idx = classes.index("Irregular") if "Irregular" in classes else 0
        features = exp.as_list(label=label_idx)
        tokens = [f[0] for f in features]
        pesos = [f[1] for f in features]
        cores = ["#d62728" if p > 0 else "#1f77b4" for p in pesos]

        axes[i].barh(tokens, pesos, color=cores)
        axes[i].axvline(0, color="black", linewidth=0.8)
        axes[i].set_title(f"Acórdão {i+1} — tokens mais preditivos de Irregular")
        axes[i].set_xlabel("Peso LIME")

    plt.tight_layout()
    caminho = FIGURAS / nome_arquivo
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Plot LIME salvo em {caminho}")
    return caminho
