"""Filtro temático e extração de label em acórdãos do TCU."""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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


def inspecionar_colunas(arquivo_csv: Path) -> list[str]:
    """Lê apenas o cabeçalho do CSV para identificar colunas disponíveis.

    Args:
        arquivo_csv: Caminho do CSV do TCU.

    Returns:
        Lista de nomes de colunas.
    """
    df_cabecalho = pd.read_csv(arquivo_csv, nrows=0, encoding="utf-8", sep=None, engine="python")
    colunas = df_cabecalho.columns.tolist()
    logger.info("Colunas encontradas (%d): %s", len(colunas), colunas)
    return colunas


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
            logger.info("Label extraída do campo 'tipo' (%.0f%% preenchidos).",
                        mapeados.notna().mean() * 100)
            return df

    # Tentativa 2: campo situacao
    if "situacao" in df.columns:
        mapeados = df["situacao"].str.extract(
            r"(Irregular|Regular com Ressalva|Regular)", expand=False
        )
        if mapeados.notna().mean() > 0.3:
            df["label"] = mapeados
            logger.info("Label extraída do campo 'situacao' (%.0f%% preenchidos).",
                        mapeados.notna().mean() * 100)
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
    logger.info("Label extraída via regex no 'sumario' (%.0f%% preenchidos).", cobertura * 100)
    return df


def filtrar_acordaos(arquivo_csv: Path, col_texto: str = "sumario") -> pd.DataFrame:
    """Carrega, filtra tematicamente e extrai labels de um CSV do TCU.

    Usa usecols para não carregar o arquivo inteiro na RAM (Guardrail 2).

    Args:
        arquivo_csv: Caminho do CSV anual do TCU.
        col_texto: Coluna usada para o filtro temático.

    Returns:
        DataFrame filtrado com coluna 'label' adicionada.
    """
    # Detectar separador e colunas disponíveis
    colunas_disponiveis = inspecionar_colunas(arquivo_csv)
    usecols = [c for c in COLUNAS_BASE if c in colunas_disponiveis]

    # Incluir urlArquivoPDF se disponível (pode ser útil para baixar PDFs depois)
    if "urlArquivoPDF" in colunas_disponiveis:
        usecols.append("urlArquivoPDF")

    logger.info("Carregando colunas: %s", usecols)
    df = pd.read_csv(
        arquivo_csv,
        usecols=usecols,
        encoding="utf-8",
        sep=None,
        engine="python",
        low_memory=False,
    )
    logger.info("Carregados %d acórdãos do arquivo %s.", len(df), arquivo_csv.name)
    logger.info("Uso de memória: %.1f MB", df.memory_usage(deep=True).sum() / 1e6)

    # Filtro temático
    col_filtro = col_texto if col_texto in df.columns else "sumario"
    df_filtrado = _filtrar_tematico(df, col_filtro)
    logger.info(
        "Após filtro temático: %d acórdãos (%.1f%% do total).",
        len(df_filtrado),
        len(df_filtrado) / len(df) * 100,
    )
    del df  # liberar memória

    # Extração de label
    df_filtrado = _extrair_label(df_filtrado)

    # Remover registros sem label
    antes = len(df_filtrado)
    df_filtrado = df_filtrado[df_filtrado["label"].notna()].copy()
    logger.info("Removidos %d registros sem label. Total final: %d.", antes - len(df_filtrado), len(df_filtrado))

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
    logger.info("Distribuição de labels:\n%s", df_total["label"].value_counts().to_string())
    return df_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filtra acórdãos do TCU por tema (saúde/educação) e extrai labels"
    )
    parser.add_argument("--anos", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--inspecionar", action="store_true",
                        help="Apenas inspeciona colunas do CSV sem filtrar")
    args = parser.parse_args()

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    if args.inspecionar:
        for ano in args.anos:
            arquivo = DATA_RAW / f"acordao-completo-{ano}.csv"
            if arquivo.exists():
                colunas = inspecionar_colunas(arquivo)
                print(f"\n{arquivo.name}: {colunas}")
    else:
        df = combinar_anos(args.anos)
        saida = DATA_INTERIM / "acordaos_filtrados.parquet"
        df.to_parquet(saida, index=False)
        print(f"\nSalvo em {saida} ({len(df)} registros)")
        print(df["label"].value_counts())
