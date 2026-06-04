"""Fine-tuning de BERTimbau para classificação de demandas de ouvidoria."""

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
MODELO_PADRAO = "neuralmind/bert-base-portuguese-cased"
RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"


def _calcular_metricas(eval_pred) -> dict:
    """Função de métricas para o Trainer — prioriza F1-macro."""
    logits, labels = eval_pred
    predicoes = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predicoes, average="macro", zero_division=0)
    return {"f1_macro": f1}


def tokenizar(
    textos: list[str],
    tokenizer,
    max_length: int = 512,
) -> dict:
    """Tokeniza uma lista de textos para entrada no modelo.

    Args:
        textos: Lista de strings.
        tokenizer: Tokenizador HuggingFace.
        max_length: Comprimento máximo de tokens.

    Returns:
        Dicionário com input_ids, attention_mask, token_type_ids.
    """
    return tokenizer(
        textos,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _preparar_dataset(
    textos: pd.Series,
    tokenizer,
    max_length: int = 512,
    rotulos: Optional[np.ndarray] = None,
) -> Dataset:
    """Converte Series de texto (e opcionalmente rótulos) em Dataset HuggingFace."""
    dados = {"texto": textos.tolist()}
    if rotulos is not None:
        dados["labels"] = rotulos.tolist()

    dataset = Dataset.from_dict(dados)

    def tokenizar_batch(batch):
        enc = tokenizer(
            batch["texto"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        if "labels" in batch:
            enc["labels"] = batch["labels"]
        return enc

    colunas_remover = ["texto"]
    dataset = dataset.map(tokenizar_batch, batched=True, remove_columns=colunas_remover)
    dataset.set_format("torch")
    return dataset


def treinar_transformer(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    X_test: pd.Series,
    modelo_nome: str = MODELO_PADRAO,
    max_length: int = 256,
    epocas: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    usar_lora: bool = False,
    seed: int = RANDOM_STATE,
) -> tuple:
    """Fine-tuning do Transformer e predição no conjunto de teste.

    Args:
        X_train: Textos de treino.
        y_train: Rótulos de treino.
        X_val: Textos de validação.
        y_val: Rótulos de validação.
        X_test: Textos de teste.
        modelo_nome: Identificador HuggingFace do modelo base.
        max_length: Comprimento máximo de tokens (≤512).
        epocas: Número de épocas de treinamento.
        batch_size: Tamanho do batch.
        lr: Taxa de aprendizado.
        usar_lora: Se True, aplica LoRA (peft) para eficiência em GPU limitada.
        seed: Semente para reprodutibilidade.

    Returns:
        (modelo_treinado, predicoes_teste, encoder_rotulos)
    """
    set_seed(seed)

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(pd.Series(y_val.index.map(lambda _: y_val.iloc[0])))
    num_rotulos = len(encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_nome, num_labels=num_rotulos
    )

    if usar_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            config_lora = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "value"],
            )
            modelo = get_peft_model(modelo, config_lora)
            modelo.print_trainable_parameters()
        except ImportError:
            print("peft não instalado — treinando modelo completo.")

    ds_treino = _preparar_dataset(X_train, tokenizer, max_length, rotulos=y_train_enc)
    ds_val = _preparar_dataset(X_val, tokenizer, max_length, rotulos=y_val_enc)

    saida_dir = RESULTADOS / "modelo_transformer"
    args_treino = TrainingArguments(
        output_dir=str(saida_dir),
        num_train_epochs=epocas,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
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

    # Predição no conjunto de teste (sem rótulos — apenas inferência)
    ds_teste = _preparar_dataset(X_test, tokenizer, max_length)
    saida_pred = trainer.predict(ds_teste)
    predicoes_enc = np.argmax(saida_pred.predictions, axis=-1)
    predicoes = encoder.inverse_transform(predicoes_enc)

    return modelo, predicoes, encoder


def carregar_modelo(
    caminho: str,
    modelo_nome: str = MODELO_PADRAO,
) -> tuple:
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
