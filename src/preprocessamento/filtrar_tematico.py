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
# O CSV real do TCU (2023) tem campos SUMARIO/VOTO que excedem 131072 bytes.
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"

# Guardrail 3: termos temáticos por domínio
# Estrutura: cada domínio é uma chave com sua lista de termos.
# O corpus principal do projeto é saude + educacao; outros domínios podem ser
# incluídos como experimento de generalização via --temas na CLI.
TERMOS_POR_TEMA: dict[str, list[str]] = {
    "saude": [
        "saúde", "saude",
        "sus", "sistema único de saúde", "sistema unico de saude",
        "fnde", "merenda", "alimentação escolar", "alimentacao escolar",
        "ministério da saúde", "ministerio da saude",
        "secretaria de saúde", "secretaria de saude",
        "hospital", "ubs", "unidade básica de saúde", "vigilância sanitária",
    ],
    "educacao": [
        "educação", "educacao",
        "mec", "ministério da educação", "ministerio da educacao",
        "secretaria de educação", "secretaria de educacao",
        "escola", "ensino fundamental", "ensino médio", "ensino medio",
        "universidade", "pnae", "pnate", "fundeb", "fundef",
    ],
    "seguranca": [
        "segurança pública", "seguranca publica",
        "polícia", "policia", "defesa civil",
        "ministério da justiça", "ministerio da justica",
        "secretaria de segurança", "secretaria de seguranca",
    ],
    "transporte": [
        "transporte", "rodovia", "infraestrutura",
        "dnit", "antt", "infraero",
        "ministério dos transportes", "ministerio dos transportes",
        "obra rodoviária", "obra rodoviaria", "pavimentação", "pavimentacao",
    ],
}

# Temas ativos por padrão (corpus principal do projeto)
TEMAS_PADRAO = ["saude", "educacao"]

# Lista plana de termos ativos (usada internamente; reconstruída em combinar_anos)
TERMOS_TEMATICOS = [
    t for tema in TEMAS_PADRAO for t in TERMOS_POR_TEMA[tema]
]


def _termos_para_temas(temas: list[str]) -> list[str]:
    """Retorna lista plana de termos para os temas solicitados."""
    termos = []
    for tema in temas:
        if tema not in TERMOS_POR_TEMA:
            raise ValueError(f"Tema desconhecido: {tema!r}. Disponíveis: {list(TERMOS_POR_TEMA)}")
        termos.extend(TERMOS_POR_TEMA[tema])
    return termos

# Colunas canônicas do projeto
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
# Normalização: lowercase, sem acentos, sem BOM/aspas/underscores/espaços
#
# Estrutura REAL do CSV do TCU (verificado em 2023-06-06):
#   Separador: pipe (|)
#   Encoding: utf-8 (sem BOM)
#   Colunas (MAIÚSCULO, com aspas no arquivo):
#     KEY, TIPO, TITULO, NUMACORDAO, ANOACORDAO, NUMATA, COLEGIADO, DATASESSAO,
#     RELATOR, SITUACAO, PROC, ACORDAOSRELACIONADOS, TIPOPROCESSO, INTERESSADOS,
#     ENTIDADE, RELATORDELIBERACAORECORRIDA, MINISTROREVISOR, MINISTROAUTORVOTOVENCEDOR,
#     REPRESENTANTEMP, UNIDADETECNICA, ADVOGADO, ASSUNTO, SUMARIO, ACORDAO,
#     DECISAO, QUORUM, MINISTROALEGOUIMPEDIMENTOSESSAO, RECURSOS,
#     RELATORIO, VOTO, DECLARACAOVOTO, VOTOCOMPLEMENTAR, VOTOMINISTROREVISOR
_MAPA_COLUNAS_NORM: dict[str, str] = {
    # numeroAcordao — campo real: NUMACORDAO
    "numacordao": "numeroAcordao",
    "numeroacordao": "numeroAcordao",
    "num_acordao": "numeroAcordao",
    "numerodoacordao": "numeroAcordao",
    "numero": "numeroAcordao",
    # anoAcordao — campo real: ANOACORDAO
    "anoacordao": "anoAcordao",
    "ano_acordao": "anoAcordao",
    "ano": "anoAcordao",
    # tipo — campo real: TIPO
    "tipo": "tipo",
    "tipodocumento": "tipo",
    "tipoacordao": "tipo",
    # situacao — campo real: SITUACAO
    "situacao": "situacao",
    "situacaodoacordao": "situacao",
    "status": "situacao",
    # sumario — campo real: SUMARIO
    "sumario": "sumario",
    "resumo": "sumario",
    "ementasumario": "sumario",
    "ementa": "sumario",
    # colegiado — campo real: COLEGIADO
    "colegiado": "colegiado",
    "orgao": "colegiado",
    "orgaocolegiado": "colegiado",
    # relator — campo real: RELATOR
    "relator": "relator",
    "ministrorelator": "relator",
    # dataSessao — campo real: DATASESSAO
    "datasessao": "dataSessao",
    "data": "dataSessao",
    "datadajulgamento": "dataSessao",
    "datajulgamento": "dataSessao",
    # voto — campo real: VOTO (texto completo do voto — D-06 atualizado!)
    "voto": "voto",
    # assunto — campo real: ASSUNTO (palavras-chave, útil para filtro temático)
    "assunto": "assunto",
    # acordao — campo real: ACORDAO (dispositivo/decisão estruturada)
    "acordao": "acordao",
    # decisao — campo real: DECISAO
    "decisao": "decisao",
    # titulo — campo real: TITULO
    "titulo": "titulo",
    # urlArquivoPDF — ausente no CSV real do TCU (voto disponível diretamente)
    "urlarquivopdf": "urlArquivoPDF",
    "url": "urlArquivoPDF",
    "urlpdf": "urlArquivoPDF",
    "linkpdf": "urlArquivoPDF",
}

# Regex para extrair label do sumario/acordao quando não há campo estruturado
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
    """Normaliza nome de coluna: strip BOM/aspas, lowercase, sem acentos/separadores."""
    nome = nome.strip().lstrip("﻿").strip('"').strip("'")
    nome = nome.lower()
    nome = unicodedata.normalize("NFD", nome)
    nome = "".join(c for c in nome if not unicodedata.combining(c))
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
    """Detecta o separador real do CSV (suporta |, ;, , e tab).

    O CSV do TCU usa pipe (|). Tenta múltiplos encodings para robustez.

    Returns:
        Separador detectado (padrão '|' se inconclusivo para CSVs do TCU).
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(arquivo_csv, "r", encoding=enc, errors="replace") as f:
                amostra = f.read(8192)
            dialect = csv.Sniffer().sniff(amostra, delimiters=",;|\t")
            return dialect.delimiter
        except csv.Error:
            continue
    return "|"  # padrão para CSVs reais do TCU


def inspecionar_colunas(arquivo_csv: Path) -> list[str]:
    """Lê apenas o cabeçalho do CSV para identificar colunas disponíveis.

    Tenta múltiplos encodings e strip de BOM/aspas dos nomes.

    Args:
        arquivo_csv: Caminho do CSV do TCU.

    Returns:
        Lista de nomes de colunas normalizados (sem BOM, aspas ou espaços extras).
    """
    sep = _detectar_separador(arquivo_csv)
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df_cab = pd.read_csv(
                arquivo_csv, nrows=0, encoding=enc, sep=sep, engine="c"
            )
            # Strip BOM, aspas e espaços dos nomes
            colunas = [c.strip().lstrip("﻿").strip('"').strip("'")
                       for c in df_cab.columns.tolist()]
            logger.info(
                "Colunas (%d) | sep=%r | enc=%s:\n  %s",
                len(colunas), sep, enc,
                "\n  ".join(colunas) if colunas else "(nenhuma)",
            )
            return colunas
        except Exception as exc:
            logger.warning("Falha ao inspecionar enc=%s: %s", enc, exc)
    return []


def _filtrar_tematico(
    df: pd.DataFrame,
    colunas_texto: list[str],
    termos: list[str] | None = None,
) -> pd.DataFrame:
    """Filtra linhas que contenham ao menos um termo temático em qualquer coluna de texto."""
    termos_ativos = termos if termos is not None else TERMOS_TEMATICOS
    padrao = "|".join(re.escape(t) for t in termos_ativos)
    mascara = pd.Series(False, index=df.index)
    for col in colunas_texto:
        if col in df.columns:
            mascara |= df[col].str.lower().str.contains(padrao, na=False)
    return df[mascara].copy()


def _extrair_regex_label(texto: str) -> str | None:
    """Extrai label via regex no texto."""
    if not isinstance(texto, str):
        return None
    m = _RE_DESFECHO.search(texto)
    if m:
        chave = m.group(1).lower().strip()
        return _MAPA_LABEL.get(chave)
    return None


def _extrair_label(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai label de desfecho do campo tipo/situacao ou via regex nos campos textuais.

    Tenta, em ordem:
    1. Campo 'tipo' (se contiver Irregular/Regular)
    2. Campo 'situacao' (se contiver desfecho)
    3. Regex no campo 'acordao' (dispositivo estruturado)
    4. Regex no campo 'sumario' (fallback)
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

    # Tentativa 3: regex no campo acordao (dispositivo)
    if "acordao" in df.columns:
        mapeados = df["acordao"].apply(_extrair_regex_label)
        cobertura = mapeados.notna().mean()
        if cobertura > 0.1:
            df["label"] = mapeados
            logger.info(
                "Label extraída via regex no campo 'acordao' (%.0f%% preenchidos).",
                cobertura * 100,
            )
            return df

    # Tentativa 4: regex no sumario (fallback final)
    col_regex = "sumario"
    if col_regex in df.columns:
        df["label"] = df[col_regex].apply(_extrair_regex_label)
        cobertura = df["label"].notna().mean()
        logger.info(
            "Label extraída via regex no 'sumario' (%.0f%% preenchidos).", cobertura * 100
        )
    else:
        df["label"] = None
        logger.error("Nenhum campo de label encontrado. Colunas: %s", df.columns.tolist())

    return df


def filtrar_acordaos(
    arquivo_csv: Path,
    col_texto: str = "sumario",
    temas: list[str] | None = None,
) -> pd.DataFrame:
    """Carrega, filtra tematicamente e extrai labels de um CSV do TCU.

    Usa usecols para não carregar o arquivo inteiro na RAM (Guardrail 2).
    Mapeia automaticamente nomes de colunas reais (ex.: NUMACORDAO, SUMARIO)
    para os nomes canônicos do projeto.

    Args:
        arquivo_csv: Caminho do CSV anual do TCU.
        col_texto: Coluna canônica usada para o filtro temático.

    Returns:
        DataFrame filtrado com coluna 'label' adicionada.
    """
    colunas_reais = inspecionar_colunas(arquivo_csv)
    sep = _detectar_separador(arquivo_csv)

    if not colunas_reais:
        raise ValueError(f"Não foi possível ler o cabeçalho de {arquivo_csv.name}")

    mapeamento = _mapear_colunas(colunas_reais)
    logger.info("Mapeamento de colunas (%d reconhecidas): %s", len(mapeamento), mapeamento)

    if not mapeamento:
        logger.error(
            "NENHUMA coluna reconhecida em %s.\n"
            "Colunas reais:\n  %s\n"
            "Adicione entradas a _MAPA_COLUNAS_NORM.",
            arquivo_csv.name,
            "\n  ".join(colunas_reais),
        )
        raise ValueError(
            f"Colunas do CSV não reconhecidas: {colunas_reais[:10]}. "
            "Veja o log e atualize _MAPA_COLUNAS_NORM."
        )

    usecols_reais = list(mapeamento.keys())

    # Carregar com múltiplos encodings até funcionar
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            logger.info(
                "Carregando %d colunas | enc=%s | sep=%r...",
                len(usecols_reais), enc, sep,
            )
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
            logger.warning("Falha ao carregar enc=%s: %s", enc, exc)

    if df is None:
        raise RuntimeError(
            f"Não foi possível carregar {arquivo_csv.name} com nenhum encoding."
        )

    # Renomear colunas reais → nomes canônicos
    df = df.rename(columns=mapeamento)
    df.columns = [c.strip().lstrip("﻿").strip('"') for c in df.columns]

    logger.info("Carregados %d acórdãos de %s.", len(df), arquivo_csv.name)
    logger.info("Uso de memória: %.1f MB", df.memory_usage(deep=True).sum() / 1e6)
    logger.info("Colunas após renomeação: %s", df.columns.tolist())

    # Filtro temático — usa sumario E assunto (se disponível)
    colunas_filtro = []
    col_canonico = col_texto if col_texto in df.columns else "sumario"
    if col_canonico in df.columns:
        colunas_filtro.append(col_canonico)
    if "assunto" in df.columns and "assunto" not in colunas_filtro:
        colunas_filtro.append("assunto")

    if not colunas_filtro:
        raise KeyError(
            f"Nenhuma coluna de texto encontrada. Disponíveis: {df.columns.tolist()}"
        )

    termos_ativos = _termos_para_temas(temas) if temas else None
    df_filtrado = _filtrar_tematico(df, colunas_filtro, termos=termos_ativos)
    logger.info(
        "Após filtro temático: %d acórdãos (%.1f%% do total).",
        len(df_filtrado),
        len(df_filtrado) / len(df) * 100 if len(df) > 0 else 0,
    )
    del df

    df_filtrado = _extrair_label(df_filtrado)

    antes = len(df_filtrado)
    df_filtrado = df_filtrado[df_filtrado["label"].notna()].copy()
    logger.info(
        "Removidos %d sem label. Total final: %d.",
        antes - len(df_filtrado),
        len(df_filtrado),
    )

    return df_filtrado


def combinar_anos(
    anos: list[int],
    destino_raw: Path = DATA_RAW,
    temas: list[str] | None = None,
) -> pd.DataFrame:
    """Combina acórdãos filtrados de múltiplos anos.

    Processa um ano por vez para não acumular CSVs brutos na RAM (Guardrail 2).

    Args:
        anos: Lista de anos a processar.
        destino_raw: Diretório com os CSVs brutos.
        temas: Lista de temas a filtrar. None = temas padrão (saude + educacao).
               Opções disponíveis: "saude", "educacao", "seguranca", "transporte".

    Returns:
        DataFrame combinado com todos os anos filtrados.
    """
    temas_efetivos = temas or TEMAS_PADRAO
    logger.info("Temas ativos: %s", temas_efetivos)
    partes = []
    for ano in anos:
        arquivo = destino_raw / f"acordao-completo-{ano}.csv"
        if not arquivo.exists():
            logger.warning("Arquivo não encontrado: %s — pulando.", arquivo)
            continue
        df_ano = filtrar_acordaos(arquivo, temas=temas_efetivos)
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
        description="Filtra acórdãos do TCU por tema e extrai labels"
    )
    parser.add_argument(
        "--anos", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024],
        help="Anos a processar (padrão: 2020-2024)",
    )
    parser.add_argument(
        "--temas", type=str, nargs="+", default=None,
        choices=list(TERMOS_POR_TEMA.keys()),
        help=f"Temas a filtrar (padrão: {TEMAS_PADRAO}). "
             f"Opções: {list(TERMOS_POR_TEMA.keys())}",
    )
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
                print(f"  Mapeamento reconhecido ({len(mapa)}): {mapa}")
    else:
        df = combinar_anos(args.anos, temas=args.temas)
        saida = DATA_INTERIM / "acordaos_filtrados.parquet"
        df.to_parquet(saida, index=False)
        print(f"\nSalvo em {saida} ({len(df)} registros)")
        print(f"Temas: {args.temas or TEMAS_PADRAO}")
        print(df["label"].value_counts())
