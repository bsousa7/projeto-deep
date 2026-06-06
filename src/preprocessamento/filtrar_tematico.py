"""Filtro temático e extração de label em acórdãos do TCU."""

import argparse
import csv
import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Aumentar limite de campo CSV para suportar textos jurídicos longos (> 128 KB)
# O CSV real do TCU tem campos sumario/voto que excedem o limite padrão de 131072 bytes.
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"

# Guardrail 3: termos temáticos obrigatórios
TERMOS_TEMATICOS = [
    "saúde",
    "sus",
    "fnde",
    "merenda",
    "educação",
    "ministério da saúde",
    "secretaria de saúde",
    "secretaria de educação",
]

# Colunas mínimas para carregar (Guardrail 2: usecols obrigatório)
# Nomes canônicos do projeto (camelCase, sem acentos)
COLUNAS_BASE = [
    "numeroAcordao",
    "anoAcordao",
    "tipo",
    "situacao",
    "sumario",
    "colegiado",
    "relator",
    "dataSessao",
]

# Mapeamento normalizado → canônico
# Normalização: lowercase, sem acentos, sem underscores/hífens/espaços, sem BOM
# Cobre variações reais do CSV do TCU (camelCase, snake_case, UPPER, acentuado)
_MAPA_COLUNAS_NORM: dict[str, str] = {
    # numeroAcordao
    "numeroacordao": "numeroAcordao",
    "num_acordao": "numeroAcordao",
    "numerodoacordao": "numeroAcordao",
    "numero": "numeroAcordao",
    # anoAcordao
    "anoacordao": "anoAcordao",
    "ano_acordao": "anoAcordao",
    "ano": "anoAcordao",
    # tipo
    "tipo": "tipo",
    "tipodocumento": "tipo",
    "tipoacordao": "tipo",
    # situacao
    "situacao": "situacao",
    "situacaodoacordao": "situacao",
    "status": "situacao",
    # sumario
    "sumario": "sumario",
    "resumo": "sumario",
    "ementasumario": "sumario",
    "ementa": "sumario",
    # colegiado
    "colegiado": "colegiado",
    "orgao": "colegiado",
    "orgaocolegiado": "colegiado",
    # relator
    "relator": "relator",
    "ministrorelator": "relator",
    # dataSessao
    "datasessao": "dataSessao",
    "data": "dataSessao",
    "datadajulgamento": "dataSessao",
    "datajulgamento": "dataSessao",
    # urlArquivoPDF
    "urlarquivopdf": "urlArquivoPDF",
    "url": "urlArquivoPDF",
    "urlpdf": "urlArquivoPDF",
    "linkpdf": "urlArquivoPDF",
}

# Regex para extrair label do sumario quando não há campo estruturado
_RE_DESFECHO = re.compile(
    r"contas\s+(irregulares|regulares\s+com\s+ressalva|regulares)",
    re.IGNORECASE,
)

# Mapeamento para label canônica
_MAPA_LABEL = {
    "irregulares": "Irregular",
    "regulares com ressalva": "Regular com Ressalva",
    "regulares": "Regular",
}


def _normalizar_nome(nome: str) -> str:
    """Normaliza nome de coluna: lowercase, sem BOM/acentos/separadores."""
    nome = nome.strip().lstrip("﻿")  # strip BOM
    nome = nome.lower()
    # remover acentos via NFD decomposition
    nome = unicodedata.normalize("NFD", nome)
    nome = "".join(c for c in nome if not unicodedata.combining(c))
    # remover underscores, hífens, espaços, pontos
    nome = re.sub(r"[\s_\-\.]", "", nome)
    return nome


def _mapear_colunas(colunas_reais: list[str]) -> dict[str, str]:
    """Retorna {nome_real: nome_canônico} para colunas que casam no mapeamento."""
    mapeamento = {}
    for col in colunas_reais:
        norm = _normalizar_nome(col)
        if norm in _MAPA_COLUNAS_NORM:
            mapeamento[col] = _MAPA_COLUNAS_NORM[norm]
    return mapeamento


def _detectar_separador(arquivo_csv: Path) -> str:
    """Lê as primeiras linhas do CSV para detectar o separador real.

    O CSV do TCU usa ponto-e-vírgula (;) ou pipe (|) como separador. Detectar
    automaticamente evita depender de sep=None + engine='python',
    que impõe um limite de 131 072 bytes por campo.

    Returns:
        Separador detectado: ',', ';', '|' ou '\t' (padrão ';' se inconclusivo).
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(arquivo_csv, "r", encoding=enc, errors="replace") as f:
                amostra = f.read(8192)
            dialect = csv.Sniffer().sniff(amostra, delimiters=",;|\t")
            logger.debug("Separador detectado com enc=%s: %r", enc, dialect.delimiter)
            return dialect.delimiter
        except csv.Error:
            continue
    return ";"


def inspecionar_colunas(arquivo_csv: Path) -> list[str]:
    """Lê apenas o cabeçalho do CSV para identificar colunas disponíveis.

    Tenta múltiplos encodings (utf-8-sig, utf-8, latin-1) e normaliza
    nomes de coluna para remover BOM e espaços extras.

    Args:
        arquivo_csv: Caminho do CSV do TCU.

    Returns:
        Lista de nomes de colunas (sem BOM, sem espaços extras).
    """
    sep = _detectar_separador(arquivo_csv)
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df_cab = pd.read_csv(
                arquivo_csv, nrows=0, encoding=enc, sep=sep, engine="c"
            )
            colunas = [c.strip().lstrip("﻿") for c in df_cab.columns.tolist()]
            logger.info(
                "Colunas (%d) | sep=%r | enc=%s:\n  %s",
                len(colunas), sep, enc,
                "\n  ".join(colunas) if colunas else "(nenhuma)",
            )
            return colunas
        except Exception as exc:
            logger.warning("Falha ao inspecionar enc=%s: %s", enc, exc)
    return []


def _filtrar_tematico(df: pd.DataFrame, col_texto: str) -> pd.DataFrame:
    """Filtra linhas que contenham ao menos um termo temático."""
    padrao = "|".join(re.escape(t) for t in TERMOS_TEMATICOS)
    mascara = df[col_texto].str.lower().str.contains(padrao, na=False)
    return df[mascara].copy()


def _extrair_label(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai label de desfecho do campo tipo/situacao ou via regex no sumario.

    Tenta, em ordem:
    1. Campo 'tipo' (se contiver Irregular/Regular)
    2. Campo 'situacao' (se contiver desfecho)
    3. Regex no 'sumario'
    """
    # Tentativa 1: campo tipo
    if "tipo" in df.columns:
        mapeados = df["tipo"].str.extract(
            r"(Irregular|Regular com Ressalva|Regular)", expand=False
        )
        if mapeados.notna().mean() > 0.3:
            df["label"] = mapeados
            logger.info(
                "Label extraída do campo 'tipo' (%.0f%% preenchidos).",
                mapeados.notna().mean() * 100,
            )
            return df

    # Tentativa 2: campo situacao
    if "situacao" in df.columns:
        mapeados = df["situacao"].str.extract(
            r"(Irregular|Regular com Ressalva|Regular)", expand=False
        )
        if mapeados.notna().mean() > 0.3:
            df["label"] = mapeados
            logger.info(
                "Label extraída do campo 'situacao' (%.0f%% preenchidos).",
                mapeados.notna().mean() * 100,
            )
            return df

    # Tentativa 3: regex no sumario
    def _extrair_regex(texto: str) -> str:
        if not isinstance(texto, str):
            return None
        m = _RE_DESFECHO.search(texto)
        if m:
            chave = m.group(1).lower().strip()
            return _MAPA_LABEL.get(chave)
        return None

    df["label"] = df["sumario"].apply(_extrair_regex)
    cobertura = df["label"].notna().mean()
    logger.info(
        "Label extraída via regex no 'sumario' (%.0f%% preenchidos).", cobertura * 100
    )
    return df


def filtrar_acordaos(arquivo_csv: Path, col_texto: str = "sumario") -> pd.DataFrame:
    """Carrega, filtra tematicamente e extrai labels de um CSV do TCU.

    Usa usecols para não carregar o arquivo inteiro na RAM (Guardrail 2).
    Mapeia automaticamente nomes de colunas reais para os nomes canônicos
    do projeto, cobrindo variações de BOM, acentos, case e separadores.

    Args:
        arquivo_csv: Caminho do CSV anual do TCU.
        col_texto: Coluna usada para o filtro temático.

    Returns:
        DataFrame filtrado com coluna 'label' adicionada.
    """
    # Detectar separador e colunas disponíveis
    colunas_reais = inspecionar_colunas(arquivo_csv)
    sep = _detectar_separador(arquivo_csv)

    if not colunas_reais:
        raise ValueError(f"Não foi possível ler o cabeçalho de {arquivo_csv.name}")

    # Mapear nomes reais → canônicos
    mapeamento = _mapear_colunas(colunas_reais)
    logger.info("Mapeamento de colunas: %s", mapeamento)

    if not mapeamento:
        # Nenhuma coluna reconhecida — mostrar colunas reais para diagnóstico
        logger.error(
            "NENHUMA coluna reconhecida em %s.\n"
            "Colunas reais encontradas:\n  %s\n"
            "Adicione entradas a _MAPA_COLUNAS_NORM para cobri-las.",
            arquivo_csv.name,
            "\n  ".join(colunas_reais),
        )
        raise ValueError(
            f"Colunas do CSV não reconhecidas: {colunas_reais[:10]}. "
            "Veja o log acima e atualize _MAPA_COLUNAS_NORM."
        )

    # Selecionar colunas reais que foram reconhecidas
    usecols_reais = list(mapeamento.keys())

    # Incluir colunas extras opcionais
    for col_extra_canonico in ("urlArquivoPDF", "texto_voto_simulado"):
        # Procurar se há coluna real que mapeia para esse canônico
        for col_r, col_c in mapeamento.items():
            if col_c == col_extra_canonico and col_r not in usecols_reais:
                usecols_reais.append(col_r)

    # Tentar múltiplos encodings para leitura completa
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            logger.info("Carregando %d colunas com enc=%s sep=%r...", len(usecols_reais), enc, sep)
            df = pd.read_csv(
                arquivo_csv,
                usecols=usecols_reais,
                encoding=enc,
                sep=sep,
                engine="c",
                on_bad_lines="warn",
            )
            break
        except Exception as exc:
            logger.warning("Falha ao carregar com enc=%s: %s", enc, exc)

    if df is None:
        raise RuntimeError(f"Não foi possível carregar {arquivo_csv.name} com nenhum encoding.")

    # Renomear colunas para nomes canônicos
    df = df.rename(columns=mapeamento)
    # Strip BOM/whitespace nos nomes de coluna que restaram
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    logger.info("Carregados %d acórdãos de %s.", len(df), arquivo_csv.name)
    logger.info("Uso de memória: %.1f MB", df.memory_usage(deep=True).sum() / 1e6)
    logger.info("Colunas após renomeação: %s", df.columns.tolist())

    # Filtro temático
    col_filtro = col_texto if col_texto in df.columns else "sumario"
    if col_filtro not in df.columns:
        disponíveis = df.columns.tolist()
        raise KeyError(
            f"Coluna de texto '{col_filtro}' não encontrada. "
            f"Colunas disponíveis: {disponíveis}"
        )

    df_filtrado = _filtrar_tematico(df, col_filtro)
    logger.info(
        "Após filtro temático: %d acórdãos (%.1f%% do total).",
        len(df_filtrado),
        len(df_filtrado) / len(df) * 100 if len(df) > 0 else 0,
    )
    del df  # liberar memória

    # Extração de label
    df_filtrado = _extrair_label(df_filtrado)

    # Remover registros sem label
    antes = len(df_filtrado)
    df_filtrado = df_filtrado[df_filtrado["label"].notna()].copy()
    logger.info(
        "Removidos %d registros sem label. Total final: %d.",
        antes - len(df_filtrado),
        len(df_filtrado),
    )

    return df_filtrado


def combinar_anos(anos: list[int], destino_raw: Path = DATA_RAW) -> pd.DataFrame:
    """Combina acórdãos filtrados de múltiplos anos.

    Processa um ano por vez para não acumular CSVs brutos na RAM (Guardrail 2).

    Args:
        anos: Lista de anos a processar.
        destino_raw: Diretório com os CSVs brutos.

    Returns:
        DataFrame combinado com todos os anos filtrados.
    """
    partes = []
    for ano in anos:
        arquivo = destino_raw / f"acordao-completo-{ano}.csv"
        if not arquivo.exists():
            logger.warning("Arquivo não encontrado: %s — pulando.", arquivo)
            continue
        df_ano = filtrar_acordaos(arquivo)
        df_ano["ano_fonte"] = ano
        partes.append(df_ano)

    if not partes:
        raise FileNotFoundError("Nenhum CSV encontrado. Execute baixar_csvs.py primeiro.")

    df_total = pd.concat(partes, ignore_index=True)
    logger.info("Total combinado: %d acórdãos.", len(df_total))
    logger.info(
        "Distribuição de labels:\n%s", df_total["label"].value_counts().to_string()
    )
    return df_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filtra acórdãos do TCU por tema (saúde/educação) e extrai labels"
    )
    parser.add_argument("--anos", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument(
        "--inspecionar",
        action="store_true",
        help="Apenas inspeciona colunas do CSV sem filtrar",
    )
    args = parser.parse_args()

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    if args.inspecionar:
        for ano in args.anos:
            arquivo = DATA_RAW / f"acordao-completo-{ano}.csv"
            if arquivo.exists():
                colunas = inspecionar_colunas(arquivo)
                mapa = _mapear_colunas(colunas)
                print(f"\n{arquivo.name}:")
                print(f"  Colunas reais ({len(colunas)}): {colunas}")
                print(f"  Mapeamento reconhecido: {mapa}")
    else:
        df = combinar_anos(args.anos)
        saida = DATA_INTERIM / "acordaos_filtrados.parquet"
        df.to_parquet(saida, index=False)
        print(f"\nSalvo em {saida} ({len(df)} registros)")
        print(df["label"].value_counts())
