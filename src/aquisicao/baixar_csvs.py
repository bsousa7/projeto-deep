"""Download dos CSVs anuais de acórdãos do TCU (Portal de Dados Abertos)."""

import argparse
import logging
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

BASE_URL = (
    "https://sites.tcu.gov.br/dados-abertos/jurisprudencia/arquivos/acordao-completo/"
    "acordao-completo-{ano}.csv"
)


def baixar_csv(ano: int, destino: Path = DATA_RAW) -> Path:
    """Baixa o CSV de acórdãos de um ano do portal TCU.

    Args:
        ano: Ano do arquivo (ex.: 2023).
        destino: Diretório de destino.

    Returns:
        Caminho do arquivo salvo.
    """
    destino.mkdir(parents=True, exist_ok=True)
    nome_arquivo = f"acordao-completo-{ano}.csv"
    caminho = destino / nome_arquivo

    if caminho.exists():
        logger.info("Arquivo já existe: %s — pulando download.", caminho)
        return caminho

    url = BASE_URL.format(ano=ano)
    logger.info("Baixando %s → %s", url, caminho)

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tamanho_total = int(resp.headers.get("content-length", 0))
        with open(caminho, "wb") as f, tqdm(
            total=tamanho_total,
            unit="B",
            unit_scale=True,
            desc=nome_arquivo,
            leave=False,
        ) as barra:
            for bloco in resp.iter_content(chunk_size=8192):
                f.write(bloco)
                barra.update(len(bloco))

    logger.info("Salvo: %s (%.1f MB)", caminho, caminho.stat().st_size / 1e6)
    return caminho


def baixar_varios(anos: list[int], destino: Path = DATA_RAW) -> list[Path]:
    """Baixa os CSVs de múltiplos anos sequencialmente.

    Args:
        anos: Lista de anos a baixar.
        destino: Diretório de destino.

    Returns:
        Lista de caminhos dos arquivos salvos.
    """
    arquivos = []
    for ano in anos:
        try:
            caminho = baixar_csv(ano, destino)
            arquivos.append(caminho)
        except requests.HTTPError as exc:
            logger.error("Falha ao baixar %d: %s", ano, exc)
    return arquivos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baixa CSVs de acórdãos do TCU (Portal de Dados Abertos)"
    )
    parser.add_argument(
        "--anos",
        type=int,
        nargs="+",
        default=[2023, 2024],
        help="Anos a baixar (padrão: 2023 2024)",
    )
    parser.add_argument(
        "--destino",
        type=Path,
        default=DATA_RAW,
        help="Diretório de destino (padrão: data/raw/)",
    )
    args = parser.parse_args()

    arquivos = baixar_varios(args.anos, args.destino)
    print(f"\nDownload concluído: {len(arquivos)} arquivo(s) em {args.destino}")
