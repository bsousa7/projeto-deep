"""Fine-tuning de LegalBert-pt para classificação de acórdãos do TCU."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

RANDOM_STATE = 42
# Modelo principal: LegalBert-pt (corpus jurídico BR)
# Fallback: neuralmind/bert-base-portuguese-cased (BERTimbau base)
MODELO_PADRAO = "dominguesm/legal-bert-base-cased-ptbr"
RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"

MAX_HEAD = 128   # tokens do início do acórdão (contexto do processo)
MAX_TAIL = 384   # tokens do final do acórdão (Dispositivo/conclusão)


def _calcular_metricas(eval_pred) -> dict:
    """Função de métricas para o Trainer — prioriza F1-macro."""
    logits, labels = eval_pred
    predicoes = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predicoes, average="macro", zero_division=0)
    return {"f1_macro": f1}


def tokenizar_head_tail(
    textos: list[str],
    tokenizer,
    max_head: int = MAX_HEAD,
    max_tail: int = MAX_TAIL,
) -> list[dict]:
    """Tokeniza cada texto aplicando estratégia head+tail.

    Concatena os primeiros max_head tokens + os últimos max_tail tokens,
    totalizando até 512 tokens por amostra.

    Referência: Sun et al. (2019) — How to Fine-Tune BERT for Text Classification.

    Args:
        textos: Lista de textos já limpos (limpar_para_bert).
        tokenizer: Tokenizador HuggingFace.
        max_head: Tokens do início.
        max_tail: Tokens do final.

    Returns:
        Lista de dicionários com input_ids e attention_mask.
    """
    encodings = []
    for texto in textos:
        ids = tokenizer.encode(texto, add_special_tokens=False)
        if len(ids) > max_head + max_tail:
            ids = ids[:max_head] + ids[-max_tail:]
        # Adicionar tokens especiais [CLS] e [SEP]
        ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        comprimento = max_head + max_tail + 2
        mascara = [1] * len(ids) + [0] * (comprimento - len(ids))
        ids = ids + [tokenizer.pad_token_id] * (comprimento - len(ids))
        encodings.append({"input_ids": ids, "attention_mask": mascara})
    return encodings


def _preparar_dataset(
    textos: pd.Series,
    tokenizer,
    max_head: int = MAX_HEAD,
    max_tail: int = MAX_TAIL,
    rotulos: Optional[np.ndarray] = None,
) -> Dataset:
    """Converte Series de texto em Dataset HuggingFace com head+tail."""
    encodings = tokenizar_head_tail(textos.tolist(), tokenizer, max_head, max_tail)

    dados = {
        "input_ids": [e["input_ids"] for e in encodings],
        "attention_mask": [e["attention_mask"] for e in encodings],
    }
    if rotulos is not None:
        dados["labels"] = rotulos.tolist()

    dataset = Dataset.from_dict(dados)
    dataset.set_format("torch")
    return dataset


def treinar_transformer(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    X_test: pd.Series,
    modelo_nome: str = MODELO_PADRAO,
    max_head: int = MAX_HEAD,
    max_tail: int = MAX_TAIL,
    epocas: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = RANDOM_STATE,
) -> tuple:
    """Fine-tuning do LegalBert-pt com truncação head+tail.

    Args:
        X_train: Textos de treino (já processados por limpar_para_bert).
        y_train: Rótulos de treino.
        X_val: Textos de validação.
        y_val: Rótulos de validação.
        X_test: Textos de teste.
        modelo_nome: Identificador HuggingFace do modelo base.
        max_head: Tokens do início do documento.
        max_tail: Tokens do final do documento.
        epocas: Número de épocas.
        batch_size: Tamanho do batch.
        lr: Taxa de aprendizado.
        seed: Semente para reprodutibilidade.

    Returns:
        (modelo_treinado, predicoes_teste, encoder_rotulos)
    """
    set_seed(seed)

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    num_rotulos = len(encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_nome, num_labels=num_rotulos
    )

    ds_treino = _preparar_dataset(X_train, tokenizer, max_head, max_tail, rotulos=y_train_enc)
    ds_val = _preparar_dataset(X_val, tokenizer, max_head, max_tail, rotulos=y_val_enc)

    saida_dir = RESULTADOS / "modelo_transformer"
    args_treino = TrainingArguments(
        output_dir=str(saida_dir),
        num_train_epochs=epocas,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=seed,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=modelo,
        args=args_treino,
        train_dataset=ds_treino,
        eval_dataset=ds_val,
        compute_metrics=_calcular_metricas,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    ds_teste = _preparar_dataset(X_test, tokenizer, max_head, max_tail)
    saida_pred = trainer.predict(ds_teste)
    predicoes_enc = np.argmax(saida_pred.predictions, axis=-1)
    predicoes = encoder.inverse_transform(predicoes_enc)

    return modelo, predicoes, encoder


def carregar_modelo(caminho: str, modelo_nome: str = MODELO_PADRAO) -> tuple:
    """Carrega modelo fine-tunado do disco.

    Args:
        caminho: Diretório com os pesos salvos.
        modelo_nome: Modelo base para o tokenizador.

    Returns:
        (modelo, tokenizador)
    """
    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModelForSequenceClassification.from_pretrained(caminho)
    return modelo, tokenizer
