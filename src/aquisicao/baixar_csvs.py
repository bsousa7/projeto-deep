"""Download dos CSVs anuais de acórdãos do TCU (Portal de Dados Abertos).

Suporta retomada de download interrompido via HTTP Range header e retry
automático com backoff exponencial.
"""

import argparse
import logging
import time
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

_MAX_TENTATIVAS = 5
_CHUNK = 1024 * 1024   # 1 MB — chunks maiores reduzem overhead de rede


def baixar_csv(ano: int, destino: Path = DATA_RAW) -> Path:
    """Baixa o CSV de acórdãos de um ano do portal TCU com suporte a retomada.

    Se o arquivo já existir parcialmente (download interrompido), retoma de
    onde parou via HTTP Range header. Faz até _MAX_TENTATIVAS com backoff
    exponencial em caso de erro de rede.

    Args:
        ano: Ano do arquivo (ex.: 2023).
        destino: Diretório de destino.

    Returns:
        Caminho do arquivo salvo.
    """
    destino.mkdir(parents=True, exist_ok=True)
    nome = f"acordao-completo-{ano}.csv"
    caminho = destino / nome
    caminho_tmp = destino / f"{nome}.part"

    url = BASE_URL.format(ano=ano)

    # Verificar tamanho real no servidor
    try:
        head = requests.head(url, timeout=30, allow_redirects=True)
        tamanho_total = int(head.headers.get("content-length", 0))
    except Exception:
        tamanho_total = 0

    # Arquivo completo já existe — verificar tamanho
    if caminho.exists():
        tamanho_local = caminho.stat().st_size
        if tamanho_total and tamanho_local >= tamanho_total:
            logger.info("Arquivo completo: %s (%.0f MB) — pulando.", nome, tamanho_local / 1e6)
            return caminho
        # Tamanho diferente — redownload completo
        logger.warning("Arquivo incompleto (%d B de %d B) — redownload.", tamanho_local, tamanho_total)
        caminho.unlink()

    for tentativa in range(1, _MAX_TENTATIVAS + 1):
        bytes_ja_baixados = caminho_tmp.stat().st_size if caminho_tmp.exists() else 0

        headers = {}
        if bytes_ja_baixados > 0:
            headers["Range"] = f"bytes={bytes_ja_baixados}-"
            logger.info(
                "Tentativa %d/%d — retomando de %.1f MB",
                tentativa, _MAX_TENTATIVAS, bytes_ja_baixados / 1e6,
            )
        else:
            logger.info("Tentativa %d/%d — baixando %s", tentativa, _MAX_TENTATIVAS, url)

        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code == 416:
                    # Range inválido — servidor não suporta ou arquivo mudou
                    caminho_tmp.unlink(missing_ok=True)
                    bytes_ja_baixados = 0
                    resp = requests.get(url, stream=True, timeout=120)

                resp.raise_for_status()

                total_barra = tamanho_total or int(resp.headers.get("content-length", 0))
                modo = "ab" if bytes_ja_baixados > 0 else "wb"

                with open(caminho_tmp, modo) as f, tqdm(
                    total=total_barra,
                    initial=bytes_ja_baixados,
                    unit="B",
                    unit_scale=True,
                    desc=nome,
                ) as barra:
                    for bloco in resp.iter_content(chunk_size=_CHUNK):
                        f.write(bloco)
                        barra.update(len(bloco))

            # Download completo — renomear arquivo temporário
            caminho_tmp.rename(caminho)
            logger.info("Salvo: %s (%.1f MB)", caminho, caminho.stat().st_size / 1e6)
            return caminho

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as exc:
            if tentativa < _MAX_TENTATIVAS:
                espera = 2 ** tentativa
                logger.warning(
                    "Erro de rede na tentativa %d: %s — aguardando %ds antes de retomar.",
                    tentativa, type(exc).__name__, espera,
                )
                time.sleep(espera)
            else:
                raise RuntimeError(
                    f"Falha após {_MAX_TENTATIVAS} tentativas ao baixar {url}.\n"
                    f"Download parcial preservado em {caminho_tmp} para retomada futura."
                ) from exc

    return caminho  # nunca chega aqui


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
        except (requests.HTTPError, RuntimeError) as exc:
            logger.error("Falha ao baixar %d: %s", ano, exc)
    return arquivos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baixa CSVs de acórdãos do TCU (Portal de Dados Abertos)"
    )
    parser.add_argument(
        "--anos", type=int, nargs="+", default=[2023, 2024],
        help="Anos a baixar (padrão: 2023 2024)",
    )
    parser.add_argument(
        "--destino", type=Path, default=DATA_RAW,
        help="Diretório de destino (padrão: data/raw/)",
    )
    args = parser.parse_args()
    arquivos = baixar_varios(args.anos, args.destino)
    print(f"\nDownload concluído: {len(arquivos)} arquivo(s) em {args.destino}")
