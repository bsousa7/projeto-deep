"""Fine-tuning de LegalBert-pt para classificação de acórdãos do TCU."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
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
MAX_TAIL = 382   # tokens do final (512 limite BERT − 2 especiais − 128 head = 382)


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


class _TrainerComPesos(Trainer):
    """Trainer com suporte a class weights para corpora desbalanceados.

    Sobrescreve compute_loss para aplicar CrossEntropyLoss ponderada,
    penalizando erros nas classes minoritárias (Irregular, Regular com Ressalva).
    """

    def __init__(self, *args, pesos_classes: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesos_classes = pesos_classes

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        peso = self.pesos_classes.to(logits.device) if self.pesos_classes is not None else None
        loss = nn.CrossEntropyLoss(weight=peso)(logits, labels)
        return (loss, outputs) if return_outputs else loss


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
    # BERT suporta no máximo 512 posições; [CLS] e [SEP] consomem 2
    max_content = 512 - 2  # = 510 tokens de conteúdo
    max_tail_efetivo = min(max_tail, max_content - max_head)  # garante ≤ 510 total
    comprimento_max = max_head + max_tail_efetivo + 2  # ≤ 512

    encodings = []
    for texto in textos:
        ids = tokenizer.encode(str(texto), add_special_tokens=False)
        if len(ids) > max_head + max_tail_efetivo:
            ids = ids[:max_head] + ids[-max_tail_efetivo:]
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
    epocas: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    seed: int = RANDOM_STATE,
    class_weights: Optional[dict] = None,
    balancear_automatico: bool = True,
) -> tuple:
    """Fine-tuning do LegalBert-pt com truncação head+tail e suporte a class weights.

    Args:
        X_train: Textos de treino (já processados por limpar_para_bert).
        y_train: Rótulos de treino.
        X_val: Textos de validação.
        y_val: Rótulos de validação.
        X_test: Textos de teste.
        modelo_nome: Identificador HuggingFace do modelo base.
        max_head: Tokens do início do documento.
        max_tail: Tokens do final do documento.
        epocas: Número de épocas (padrão 5 para corpus pequeno).
        batch_size: Tamanho do batch.
        lr: Taxa de aprendizado (padrão 1e-5, conservador para corpus pequeno).
        seed: Semente para reprodutibilidade.
        class_weights: Dict {classe: peso} para corrigir desbalanceamento.
                       Ex.: {"Irregular": 2.1, "Regular": 0.7, "Regular com Ressalva": 3.5}
                       Se None e balancear_automatico=True, calcula automaticamente.
        balancear_automatico: Se True e class_weights=None, calcula pesos via
                              sklearn compute_class_weight('balanced').

    Returns:
        (modelo_treinado, predicoes_teste, encoder_rotulos)
    """
    set_seed(seed)

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    num_rotulos = len(encoder.classes_)

    # Calcular / converter class weights
    pesos_tensor: Optional[torch.Tensor] = None
    if class_weights is not None:
        # Garantir ordem igual ao LabelEncoder
        pesos = [class_weights[c] for c in encoder.classes_]
        pesos_tensor = torch.tensor(pesos, dtype=torch.float)
        print(f"Class weights (manual): { {c: round(p,3) for c,p in zip(encoder.classes_, pesos)} }")
    elif balancear_automatico:
        pesos = compute_class_weight("balanced", classes=encoder.classes_, y=y_train)
        pesos_tensor = torch.tensor(pesos, dtype=torch.float)
        print(f"Class weights (auto): { {c: round(p,3) for c,p in zip(encoder.classes_, pesos)} }")
    else:
        print("Class weights: desativado")

    print(f"Classes: {list(encoder.classes_)}")
    print(f"Distribuição treino: { {c: int((y_train==c).sum()) for c in encoder.classes_} }")
    print(f"Modelo: {modelo_nome}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ausente'}")

    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_nome, num_labels=num_rotulos, ignore_mismatched_sizes=True
    )

    ds_treino = _preparar_dataset(X_train, tokenizer, max_head, max_tail, rotulos=y_train_enc)
    ds_val = _preparar_dataset(X_val, tokenizer, max_head, max_tail, rotulos=y_val_enc)

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

    trainer = _TrainerComPesos(
        model=modelo,
        args=args_treino,
        train_dataset=ds_treino,
        eval_dataset=ds_val,
        compute_metrics=_calcular_metricas,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        pesos_classes=pesos_tensor,
    )

    print(f"\nIniciando fine-tuning | {epocas} épocas | batch={batch_size} | lr={lr}")
    trainer.train()

    ds_teste = _preparar_dataset(X_test, tokenizer, max_head, max_tail)
    saida_pred = trainer.predict(ds_teste)
    predicoes_enc = np.argmax(saida_pred.predictions, axis=-1)
    predicoes = encoder.inverse_transform(predicoes_enc)

    return modelo, predicoes, encoder


def treinar_com_lora(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    X_test: pd.Series,
    modelo_nome: str = MODELO_PADRAO,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    epocas: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    seed: int = RANDOM_STATE,
    max_head: int = MAX_HEAD,
    max_tail: int = MAX_TAIL,
) -> tuple:
    """Fine-tuning com LoRA (PEFT) — viabiliza K-Fold no T4 (Aula 08).

    Congela os 110M parâmetros do LegalBert-pt e injeta adaptadores LoRA
    de baixo posto nas camadas de atenção, reduzindo parâmetros treináveis
    para ~300K (−99,7%). Isso permite executar K-Fold (5 folds) com custo
    comparável a um único treinamento completo (Hu et al., 2022).

    Args:
        X_train, y_train: Dados de treino.
        X_val, y_val: Dados de validação (early stopping).
        X_test: Dados de teste para predição final.
        modelo_nome: Identificador HuggingFace do modelo base.
        lora_r: Rank das matrizes de baixo posto (padrão=8).
        lora_alpha: Escala da atualização LoRA (padrão=16).
        lora_dropout: Dropout nas camadas LoRA (padrão=0.1).
        epocas: Épocas de treinamento.
        batch_size: Tamanho do batch.
        lr: Taxa de aprendizado (maior que Full FT por ser PEFT).
        seed: Semente aleatória.
        max_head: Tokens do início (contexto do processo).
        max_tail: Tokens do final (dispositivo).

    Returns:
        (modelo_lora, predicoes, encoder)
    """
    from peft import LoraConfig, TaskType, get_peft_model

    set_seed(seed)
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    num_rotulos = len(encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    base = AutoModelForSequenceClassification.from_pretrained(
        modelo_nome, num_labels=num_rotulos, ignore_mismatched_sizes=True
    )

    config_lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],
        bias="none",
    )
    modelo = get_peft_model(base, config_lora)

    params_treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    params_total      = sum(p.numel() for p in modelo.parameters())
    print(f"\nLoRA ativo: {params_treinaveis:,} parâmetros treináveis "
          f"de {params_total:,} totais "
          f"({100 * params_treinaveis / params_total:.2f}%)")

    pesos = compute_class_weight("balanced", classes=encoder.classes_, y=y_train)
    pesos_tensor = torch.tensor(pesos, dtype=torch.float)
    print(f"Class weights: { {c: round(p, 3) for c, p in zip(encoder.classes_, pesos)} }")

    ds_treino = _preparar_dataset(X_train, tokenizer, max_head, max_tail, rotulos=y_train_enc)
    ds_val    = _preparar_dataset(X_val,   tokenizer, max_head, max_tail, rotulos=y_val_enc)

    steps_por_epoca = max(1, len(X_train) // batch_size)
    warmup_steps    = max(1, int(steps_por_epoca * epocas * 0.1))

    saida_dir = RESULTADOS / "modelo_lora"
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

    trainer = _TrainerComPesos(
        model=modelo,
        args=args_treino,
        train_dataset=ds_treino,
        eval_dataset=ds_val,
        compute_metrics=_calcular_metricas,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        pesos_classes=pesos_tensor,
    )

    print(f"\nIniciando LoRA fine-tuning | {epocas} épocas | batch={batch_size} | lr={lr}")
    trainer.train()

    ds_teste = _preparar_dataset(X_test, tokenizer, max_head, max_tail)
    saida_pred = trainer.predict(ds_teste)
    predicoes_enc = np.argmax(saida_pred.predictions, axis=-1)
    predicoes = encoder.inverse_transform(predicoes_enc)

    return modelo, predicoes, encoder


def kfold_com_lora(
    X: pd.Series,
    y: pd.Series,
    modelo_nome: str = MODELO_PADRAO,
    n_splits: int = 5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    seed: int = RANDOM_STATE,
    epocas: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    max_head: int = MAX_HEAD,
    max_tail: int = MAX_TAIL,
) -> dict:
    """K-Fold estratificado com LoRA — viável no T4 por reinicializar só adaptadores.

    Cada fold: congela backbone, inicializa adaptadores LoRA frescos, treina,
    avalia. Custo total ≈ 1 treinamento completo × n_splits (vs. 5× para Full FT).

    Returns:
        Dicionário com f1_macro_por_fold, média, std e IC 95%.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.model_selection import StratifiedKFold

    set_seed(seed)
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    num_rotulos = len(encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1_folds = []

    for fold, (idx_treino, idx_val) in enumerate(kf.split(X, y_enc)):
        print(f"\n{'='*60}\nFold {fold+1}/{n_splits}")

        X_tr, X_v = X.iloc[idx_treino], X.iloc[idx_val]
        y_tr, y_v = y_enc[idx_treino], y_enc[idx_val]

        base = AutoModelForSequenceClassification.from_pretrained(
            modelo_nome, num_labels=num_rotulos, ignore_mismatched_sizes=True
        )
        config_lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["query", "value"], bias="none",
        )
        modelo_fold = get_peft_model(base, config_lora)

        pesos = compute_class_weight("balanced", classes=encoder.classes_, y=y_tr)
        pesos_tensor = torch.tensor(pesos, dtype=torch.float)

        ds_tr = _preparar_dataset(X_tr, tokenizer, max_head, max_tail, rotulos=y_tr)
        ds_v  = _preparar_dataset(X_v,  tokenizer, max_head, max_tail, rotulos=y_v)

        saida_dir = RESULTADOS / f"lora_fold_{fold}"
        saida_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(saida_dir),
            num_train_epochs=epocas,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr, weight_decay=0.01,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro", greater_is_better=True,
            seed=seed + fold, fp16=torch.cuda.is_available(),
            logging_steps=50, report_to="none", save_total_limit=1,
        )
        trainer = _TrainerComPesos(
            model=modelo_fold, args=args,
            train_dataset=ds_tr, eval_dataset=ds_v,
            compute_metrics=_calcular_metricas,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
            pesos_classes=pesos_tensor,
        )
        trainer.train()

        saida = trainer.predict(ds_v)
        pred_enc = np.argmax(saida.predictions, axis=-1)
        f1 = f1_score(y_v, pred_enc, average="macro", zero_division=0)
        f1_folds.append(f1)
        print(f"  Fold {fold+1} F1-macro: {f1:.4f}")

    f1_arr = np.array(f1_folds)
    print(f"\n{'='*60}")
    print(f"K-Fold LoRA ({n_splits} folds) — {modelo_nome}")
    print(f"  F1-macro por fold: {[round(f, 4) for f in f1_arr]}")
    print(f"  Média: {f1_arr.mean():.4f}  ±  {f1_arr.std():.4f}")
    print(f"  IC 95%: [{f1_arr.mean()-2*f1_arr.std():.4f}, {f1_arr.mean()+2*f1_arr.std():.4f}]")

    return {
        "modelo": modelo_nome,
        "metodo": "LoRA",
        "lora_r": lora_r,
        "n_splits": n_splits,
        "f1_macro_por_fold": f1_arr.tolist(),
        "f1_macro_media": round(float(f1_arr.mean()), 4),
        "f1_macro_std":   round(float(f1_arr.std()),  4),
        "ic95_lower": round(float(f1_arr.mean() - 2 * f1_arr.std()), 4),
        "ic95_upper": round(float(f1_arr.mean() + 2 * f1_arr.std()), 4),
    }


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
