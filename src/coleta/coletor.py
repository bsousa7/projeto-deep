"""Coleta de manifestações de ouvidoria via scraping/API."""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "OuvidoriaNLP-Pesquisa-IDP/1.0 "
        "(Trabalho acadêmico; contato: bruno.aires9@gmail.com)"
    )
}

# Padrões para anonimização
_RE_CPF = re.compile(r"\b\d{3}[.\-]?\d{3}[.\-]?\d{3}[-]?\d{2}\b")
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_RE_TELEFONE = re.compile(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b")


def anonimizar(texto: str) -> str:
    """Remove CPF, e-mail e telefone de um texto."""
    texto = _RE_CPF.sub("[CPF]", texto)
    texto = _RE_EMAIL.sub("[EMAIL]", texto)
    texto = _RE_TELEFONE.sub("[FONE]", texto)
    return texto


def _salvar_log(fonte: str, total: int, arquivo: Path) -> None:
    """Grava log de coleta com timestamp e contagem."""
    registro = {
        "fonte": fonte,
        "total_registros": total,
        "arquivo_saida": str(arquivo),
        "timestamp": datetime.now().isoformat(),
    }
    log_path = DATA_RAW / f"log_coleta_{datetime.now().strftime('%Y-%m-%d')}.json"
    historico = []
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            historico = json.load(f)
    historico.append(registro)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=2)
    logger.info("Log gravado em %s", log_path)


def coletar_consumidor(
    segmento: str = "",
    paginas: int = 10,
    delay: float = 2.0,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Coleta relatos do Consumidor.gov.br via interface de busca.

    Args:
        segmento: Filtro de segmento/setor (ex.: 'Telecomunicações').
        paginas: Número de páginas a percorrer.
        delay: Intervalo em segundos entre requisições.
        dry_run: Se True, retorna DataFrame vazio sem fazer requisições.

    Returns:
        DataFrame com colunas: texto, categoria, data_coleta.
    """
    if dry_run:
        logger.info("[dry_run] Nenhuma requisição disparada.")
        return pd.DataFrame(columns=["texto", "categoria", "data_coleta"])

    BASE_URL = "https://www.consumidor.gov.br/pages/relatos/buscar"
    registros = []

    for pagina in range(1, paginas + 1):
        params = {"pagina": pagina}
        if segmento:
            params["segmento"] = segmento

        try:
            resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Erro na página %d: %s", pagina, exc)
            time.sleep(delay * 2)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        itens = soup.select(".relato-item")

        if not itens:
            logger.info("Sem mais itens na página %d.", pagina)
            break

        for item in itens:
            texto_elem = item.select_one(".descricao-relato")
            cat_elem = item.select_one(".assunto-relato")
            if not texto_elem:
                continue
            texto = anonimizar(texto_elem.get_text(strip=True))
            categoria = cat_elem.get_text(strip=True) if cat_elem else ""
            registros.append(
                {"texto": texto, "categoria": categoria, "data_coleta": datetime.now().date().isoformat()}
            )

        logger.info("Página %d: %d registros acumulados", pagina, len(registros))
        time.sleep(delay)

    df = pd.DataFrame(registros)
    return df


def coletar_falabr(
    paginas: int = 10,
    delay: float = 2.0,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Coleta manifestações públicas do Fala.BR (CGU).

    Args:
        paginas: Número de páginas a percorrer.
        delay: Intervalo em segundos entre requisições.
        dry_run: Se True, retorna DataFrame vazio sem fazer requisições.

    Returns:
        DataFrame com colunas: texto, categoria, data_coleta.
    """
    if dry_run:
        logger.info("[dry_run] Nenhuma requisição disparada.")
        return pd.DataFrame(columns=["texto", "categoria", "data_coleta"])

    BASE_URL = "https://falabr.cgu.gov.br/publico/Manifestacao/SelecionarTipoManifestacao.aspx"
    registros = []

    for pagina in range(1, paginas + 1):
        params = {"pagina": pagina}
        try:
            resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Erro na página %d: %s", pagina, exc)
            time.sleep(delay * 2)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        itens = soup.select(".manifestacao-item, .list-item")

        if not itens:
            logger.info("Sem mais itens na página %d.", pagina)
            break

        for item in itens:
            texto_elem = item.select_one(".texto-manifestacao, .descricao")
            cat_elem = item.select_one(".tipo-manifestacao, .categoria")
            if not texto_elem:
                continue
            texto = anonimizar(texto_elem.get_text(strip=True))
            categoria = cat_elem.get_text(strip=True) if cat_elem else ""
            registros.append(
                {"texto": texto, "categoria": categoria, "data_coleta": datetime.now().date().isoformat()}
            )

        logger.info("Página %d: %d registros acumulados", pagina, len(registros))
        time.sleep(delay)

    df = pd.DataFrame(registros)
    return df


def coletar(
    fonte: str = "consumidor",
    paginas: int = 10,
    delay: float = 2.0,
    dry_run: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Ponto de entrada unificado para coleta de dados.

    Args:
        fonte: 'consumidor' ou 'falabr'.
        paginas: Quantidade de páginas a percorrer.
        delay: Intervalo entre requisições (segundos).
        dry_run: Se True, retorna DataFrame vazio sem fazer requisições.
        **kwargs: Parâmetros adicionais repassados à função específica.

    Returns:
        DataFrame com colunas: texto, categoria, data_coleta.
    """
    fontes_disponiveis = {"consumidor": coletar_consumidor, "falabr": coletar_falabr}
    if fonte not in fontes_disponiveis:
        raise ValueError(f"Fonte inválida: '{fonte}'. Use: {list(fontes_disponiveis)}")

    logger.info("Iniciando coleta — fonte=%s, paginas=%d, dry_run=%s", fonte, paginas, dry_run)
    df = fontes_disponiveis[fonte](paginas=paginas, delay=delay, dry_run=dry_run, **kwargs)

    if df.empty:
        logger.warning("Coleta retornou DataFrame vazio.")
        return df

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    saida = DATA_RAW / "coleta.parquet"
    df.to_parquet(saida, index=False)
    logger.info("Dados salvos em %s (%d registros)", saida, len(df))
    _salvar_log(fonte, len(df), saida)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coleta de demandas de ouvidoria")
    parser.add_argument("--fonte", default="consumidor", choices=["consumidor", "falabr"])
    parser.add_argument("--paginas", type=int, default=5)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = coletar(
        fonte=args.fonte,
        paginas=args.paginas,
        delay=args.delay,
        dry_run=args.dry_run,
    )
    print(df.head())
    print(f"\nTotal coletado: {len(df)} registros")
