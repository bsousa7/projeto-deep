"""Fine-tuning de LegalBert-pt para classificação de acórdãos do TCU."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
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
MODELO_PADRAO = "dominguesm/legal-bert-base-cased-ptbr"
RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"

MAX_HEAD = 128   # tokens do início do acórdão (contexto do processo)
MAX_TAIL = 384   # tokens do final do acórdão (Dispositivo/conclusão)


class _TorchDataset(TorchDataset):
    """Dataset PyTorch puro.

    Substitui datasets.Dataset.set_format('torch') que causa
    ImportError: cannot import name 'VideoReader' from torchvision.io
    em versões recentes do torchvision no Colab.
    """

    def __init__(self, encodings: list[dict], rotulos: Optional[np.ndarray] = None):
        self.input_ids = [e["input_ids"] for e in encodings]
        self.attention_mask = [e["attention_mask"] for e in encodings]
        self.rotulos = rotulos

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        if self.rotulos is not None:
            item["labels"] = torch.tensor(int(self.rotulos[idx]), dtype=torch.long)
        return item


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
        textos: Lista de textos já limpos.
        tokenizer: Tokenizador HuggingFace.
        max_head: Tokens do início.
        max_tail: Tokens do final.

    Returns:
        Lista de dicionários com input_ids e attention_mask.
    """
    comprimento_max = max_head + max_tail + 2  # +2 para [CLS] e [SEP]
    encodings = []
    for texto in textos:
        ids = tokenizer.encode(str(texto), add_special_tokens=False)
        if len(ids) > max_head + max_tail:
            ids = ids[:max_head] + ids[-max_tail:]
        ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        n_pad = comprimento_max - len(ids)
        mascara = [1] * len(ids) + [0] * n_pad
        ids = ids + [tokenizer.pad_token_id] * n_pad
        encodings.append({"input_ids": ids, "attention_mask": mascara})
    return encodings


def _preparar_dataset(
    textos: pd.Series,
    tokenizer,
    max_head: int = MAX_HEAD,
    max_tail: int = MAX_TAIL,
    rotulos: Optional[np.ndarray] = None,
) -> _TorchDataset:
    """Converte Series de texto em _TorchDataset com tokenização head+tail."""
    encodings = tokenizar_head_tail(textos.tolist(), tokenizer, max_head, max_tail)
    return _TorchDataset(encodings, rotulos)


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

    print(f"Classes: {list(encoder.classes_)}")
    print(f"Modelo: {modelo_nome}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ausente'}")

    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_nome, num_labels=num_rotulos, ignore_mismatched_sizes=True
    )

    ds_treino = _preparar_dataset(X_train, tokenizer, max_head, max_tail, rotulos=y_train_enc)
    ds_val = _preparar_dataset(X_val, tokenizer, max_head, max_tail, rotulos=y_val_enc)

    # warmup_steps: 10% do total de steps de treino
    steps_por_epoca = max(1, len(X_train) // batch_size)
    warmup_steps = max(1, int(steps_por_epoca * epocas * 0.1))

    saida_dir = RESULTADOS / "modelo_transformer"
    saida_dir.mkdir(parents=True, exist_ok=True)

    args_treino = TrainingArguments(
        output_dir=str(saida_dir),
        num_train_epochs=epocas,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=seed,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=modelo,
        args=args_treino,
        train_dataset=ds_treino,
        eval_dataset=ds_val,
        compute_metrics=_calcular_metricas,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\nIniciando fine-tuning | {epocas} épocas | batch={batch_size} | lr={lr}")
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
