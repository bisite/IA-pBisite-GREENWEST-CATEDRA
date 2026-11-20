#!/usr/bin/env python3
"""
descarga_ifn.py
===============
Descarga masiva de datos del Inventario Forestal Nacional (IFN2, IFN3, IFN4)
desde el servidor del MITECO.

Uso:
    python descarga_ifn.py [--dest CARPETA] [--workers N] [--inventarios ifn2 ifn3 ifn4]

Opciones:
    --dest          Carpeta raíz de destino (por defecto: IFN_España)
    --workers       Descargas en paralelo (por defecto: 4)
    --inventarios   Qué inventarios descargar: ifn2, ifn3, ifn4 (por defecto: los tres)

Ejemplo:
    python descarga_ifn.py --dest /datos/IFN_España --workers 6
    python descarga_ifn.py --inventarios ifn4
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("ERROR: Instala la dependencia: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("descarga_ifn.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# URLs de descarga
# ──────────────────────────────────────────────────────────────────────────────
_BDN = "https://www.miteco.gob.es/content/dam/miteco/es/biodiversidad/servicios/banco-datos-naturaleza"
_INV = "https://www.miteco.gob.es/content/dam/miteco/es/biodiversidad/temas/inventarios-nacionales"

# IFN2 — Ficheros de parcelas y proceso de datos (DBF en ZIP)
# Fuente: https://www.miteco.gob.es/es/biodiversidad/servicios/banco-datos-naturaleza/informacion-disponible/ifn2_descargas.html
IFN2_URLS = [
    (f"{_BDN}/090471228013cbb1_tcm30-278506.zip", "IFN2"),  # 01 Álava
    (f"{_BDN}/090471228013cbb5_tcm30-278507.zip", "IFN2"),  # 02 Albacete
    (f"{_BDN}/090471228013cbb7_tcm30-278508.zip", "IFN2"),  # 03 Alicante
    (f"{_BDN}/090471228013cbb9_tcm30-278509.zip", "IFN2"),  # 04 Almería
    (f"{_BDN}/090471228013cbbb_tcm30-278510.zip", "IFN2"),  # 05 Ávila
    (f"{_BDN}/090471228013cbbd_tcm30-278511.zip", "IFN2"),  # 06 Badajoz
    (f"{_BDN}/090471228013cbbf_tcm30-278512.zip", "IFN2"),  # 07 Baleares
    (f"{_BDN}/090471228013cbc1_tcm30-278513.zip", "IFN2"),  # 08 Barcelona
    (f"{_BDN}/090471228013cbc3_tcm30-278514.zip", "IFN2"),  # 09 Burgos
    (f"{_BDN}/090471228013cbc5_tcm30-278515.zip", "IFN2"),  # 10 Cáceres
    (f"{_BDN}/090471228013cbc7_tcm30-278516.zip", "IFN2"),  # 11 Cádiz
    (f"{_BDN}/090471228013cbee_tcm30-278517.zip", "IFN2"),  # 12 Castellón
    (f"{_BDN}/090471228013cbf2_tcm30-278518.zip", "IFN2"),  # 13 Ciudad Real
    (f"{_BDN}/090471228013cbf4_tcm30-278519.zip", "IFN2"),  # 14 Córdoba
    (f"{_BDN}/090471228013cbf8_tcm30-278520.zip", "IFN2"),  # 15 A Coruña
    (f"{_BDN}/090471228013cbfc_tcm30-278521.zip", "IFN2"),  # 16 Cuenca
    (f"{_BDN}/090471228013cc00_tcm30-278522.zip", "IFN2"),  # 17 Girona
    (f"{_BDN}/090471228013cc04_tcm30-278523.zip", "IFN2"),  # 18 Granada
    (f"{_BDN}/090471228013cc06_tcm30-278524.zip", "IFN2"),  # 19 Guadalajara
    (f"{_BDN}/090471228013cc0a_tcm30-278525.zip", "IFN2"),  # 20 Guipúzcoa
    (f"{_BDN}/090471228013cc0e_tcm30-278526.zip", "IFN2"),  # 21 Huelva
    (f"{_BDN}/090471228013cc52_tcm30-278527.zip", "IFN2"),  # 22 Huesca
    (f"{_BDN}/090471228013cc54_tcm30-278528.zip", "IFN2"),  # 23 Jaén
    (f"{_BDN}/090471228013cc56_tcm30-278529.zip", "IFN2"),  # 24 León
    (f"{_BDN}/090471228013cc58_tcm30-278530.zip", "IFN2"),  # 25 Lleida
    (f"{_BDN}/090471228013cc5a_tcm30-278531.zip", "IFN2"),  # 26 La Rioja
    (f"{_BDN}/090471228013cc5c_tcm30-278532.zip", "IFN2"),  # 27 Lugo
    (f"{_BDN}/090471228013cc5e_tcm30-278533.zip", "IFN2"),  # 28 Madrid
    (f"{_BDN}/090471228013cc60_tcm30-278534.zip", "IFN2"),  # 29 Málaga
    (f"{_BDN}/090471228013cc62_tcm30-278535.zip", "IFN2"),  # 30 Murcia
    (f"{_BDN}/090471228013cc64_tcm30-278536.zip", "IFN2"),  # 31 Navarra
    (f"{_BDN}/090471228013cc66_tcm30-278537.zip", "IFN2"),  # 32 Ourense
    (f"{_BDN}/090471228013cc68_tcm30-278538.zip", "IFN2"),  # 33 Asturias
    (f"{_BDN}/090471228013cc6a_tcm30-278539.zip", "IFN2"),  # 34 Palencia
    (f"{_BDN}/090471228013cc6c_tcm30-278540.zip", "IFN2"),  # 35 Las Palmas
    (f"{_BDN}/090471228013cc6e_tcm30-278541.zip", "IFN2"),  # 36 Pontevedra
    (f"{_BDN}/090471228013cc70_tcm30-278542.zip", "IFN2"),  # 37 Salamanca
    (f"{_BDN}/090471228013cc72_tcm30-278543.zip", "IFN2"),  # 38 S.C. Tenerife
    (f"{_BDN}/090471228013cc74_tcm30-278544.zip", "IFN2"),  # 39 Cantabria
    (f"{_BDN}/090471228013cc76_tcm30-278545.zip", "IFN2"),  # 40 Segovia
    (f"{_BDN}/090471228013cc78_tcm30-278546.zip", "IFN2"),  # 41 Sevilla
    (f"{_BDN}/090471228013cc7a_tcm30-278547.zip", "IFN2"),  # 42 Soria
    (f"{_BDN}/090471228013cc7c_tcm30-278548.zip", "IFN2"),  # 43 Tarragona
    (f"{_BDN}/090471228013cc7e_tcm30-278549.zip", "IFN2"),  # 44 Teruel
    (f"{_BDN}/090471228013cc80_tcm30-278550.zip", "IFN2"),  # 45 Toledo
    (f"{_BDN}/090471228013cc82_tcm30-278551.zip", "IFN2"),  # 46 Valencia
    (f"{_BDN}/090471228013cc84_tcm30-278552.zip", "IFN2"),  # 47 Valladolid
    (f"{_BDN}/090471228013cc86_tcm30-278553.zip", "IFN2"),  # 48 Vizcaya
    (f"{_BDN}/090471228013cc88_tcm30-278554.zip", "IFN2"),  # 49 Zamora
    (f"{_BDN}/090471228013cc8a_tcm30-278555.zip", "IFN2"),  # 50 Zaragoza
]

# IFN3 — Campo (.accdb en ZIP, patrón uniforme por número de provincia)
# Fuente: https://www.miteco.gob.es/es/biodiversidad/servicios/banco-datos-naturaleza/informacion-disponible/ifn3_base_datos_1_25.html
IFN3_CAMPO_URLS = [
    (f"{_BDN}/Ifn3p{i:02d}.zip", "IFN3_campo") for i in range(1, 51)
]

# IFN3 — SIG (.accdb en ZIP, patrón uniforme por número de provincia)
IFN3_SIG_URLS = [
    (f"{_BDN}/Sig_{i:02d}.zip", "IFN3_sig") for i in range(1, 51)
]

# IFN4 — Campo (.accdb en ZIP, sin patrón uniforme — nombres irregulares)
# Nota: algunas CC.AA. tienen un fichero unificado (Canarias, Cataluña, Extremadura, País Vasco)
# Fuente: https://www.miteco.gob.es/es/biodiversidad/temas/inventarios-nacionales/inventario-forestal-nacional/cuarto_inventario.html
IFN4_CAMPO_URLS = [
    (f"{_INV}/Ifn4p15-coruna.accdb.zip",                   "IFN4_campo"),  # A Coruña
    (f"{_INV}/ifn/ifn4/ifn4-albacete.zip",                 "IFN4_campo"),  # Albacete
    (f"{_INV}/Ifn4p03_Alicante.zip",                       "IFN4_campo"),  # Alicante
    (f"{_INV}/Ifn4p33-asturias.accdb.zip",                 "IFN4_campo"),  # Asturias
    (f"{_INV}/Ifn4p05-avila.accdb.zip",                    "IFN4_campo"),  # Ávila
    (f"{_INV}/Ifn4p07-baleares.accdb.zip",                 "IFN4_campo"),  # Baleares
    (f"{_INV}/Ifn4p09-burgos.accdb.zip",                   "IFN4_campo"),  # Burgos
    (f"{_INV}/2025_12_Ifn4p11.accdb.zip",                  "IFN4_campo"),  # Cádiz
    (f"{_INV}/ifn/ifn4/ifn4_canarias_tcm30-536601.zip",    "IFN4_campo"),  # Canarias (Las Palmas + Tenerife)
    (f"{_INV}/Ifn4p39-cantabria.accdb.zip",                "IFN4_campo"),  # Cantabria
    (f"{_INV}/Ifn4p12_Castell%C3%B3n.zip",                 "IFN4_campo"),  # Castellón
    (f"{_INV}/ifn/ifn4/ifn4_cataluna_tcm30-536603.zip",    "IFN4_campo"),  # Cataluña (Barcelona+Girona+Lleida+Tarragona)
    (f"{_INV}/ifn/ifn4/ifn4-ciudadreal.zip",               "IFN4_campo"),  # Ciudad Real
    (f"{_INV}/2025_12_Ifn4p14.accdb.zip",                  "IFN4_campo"),  # Córdoba
    (f"{_INV}/ifn/ifn4/ifn4-cuenca.zip",                   "IFN4_campo"),  # Cuenca
    (f"{_INV}/ifn/ifn4/ifn4_extremadura_tcm30-536579.zip", "IFN4_campo"),  # Extremadura (Badajoz+Cáceres)
    (f"{_INV}/ifn/ifn4/ifn4-guadalajara.zip",              "IFN4_campo"),  # Guadalajara
    (f"{_INV}/Ifn4p22_Huesca.zip",                         "IFN4_campo"),  # Huesca
    (f"{_INV}/2025_12_ifn4p23.accdb.zip",                  "IFN4_campo"),  # Jaén
    (f"{_INV}/Ifn4p24-leon.accdb.zip",                     "IFN4_campo"),  # León
    (f"{_INV}/Ifn4p26-rioja.accdb.zip",                    "IFN4_campo"),  # La Rioja
    (f"{_INV}/Ifn4p27-lugo.accdb.zip",                     "IFN4_campo"),  # Lugo
    (f"{_INV}/Ifn4p28-madrid.accdb.zip",                   "IFN4_campo"),  # Madrid
    (f"{_INV}/2025_12_Ifn4p29.accdb.zip",                  "IFN4_campo"),  # Málaga
    (f"{_INV}/Ifn4p30-murcia.accdb.zip",                   "IFN4_campo"),  # Murcia
    (f"{_INV}/Ifn4p31-navarra.accdb.zip",                  "IFN4_campo"),  # Navarra
    (f"{_INV}/Ifn4p32-ourense.accdb.zip",                  "IFN4_campo"),  # Ourense
    (f"{_INV}/Ifn4p-paisvasco.accdb.zip",                  "IFN4_campo"),  # País Vasco (Álava+Guipúzcoa+Vizcaya)
    (f"{_INV}/Ifn4p34-palencia.accdb.zip",                 "IFN4_campo"),  # Palencia
    (f"{_INV}/Ifn4p36-pontevedra.accdb.zip",               "IFN4_campo"),  # Pontevedra
    (f"{_INV}/Ifn4p37-salamanca.accdb.zip",                "IFN4_campo"),  # Salamanca
    (f"{_INV}/Ifn4p40-segovia.accdb.zip",                  "IFN4_campo"),  # Segovia
    (f"{_INV}/Ifn4p42-soria.accdb.zip",                    "IFN4_campo"),  # Soria
    (f"{_INV}/Ifn4p44_Teruel.zip",                         "IFN4_campo"),  # Teruel
    (f"{_INV}/ifn/ifn4/ifn4-toledo.zip",                   "IFN4_campo"),  # Toledo
    (f"{_INV}/Ifn4p46_Valencia.zip",                       "IFN4_campo"),  # Valencia
    (f"{_INV}/Ifn4p47-valladolid.accdb.zip",               "IFN4_campo"),  # Valladolid
    (f"{_INV}/Ifn4p49-zamora.accdb.zip",                   "IFN4_campo"),  # Zamora
    (f"{_INV}/Ifn4p50_Zaragoza.zip",                       "IFN4_campo"),  # Zaragoza
]

# IFN4 — SIG (bases de datos SIG, mismas agrupaciones que campo)
IFN4_SIG_URLS = [
    (f"{_INV}/sig_acoruna_tcm30-536623.zip",                "IFN4_sig"),  # A Coruña
    (f"{_INV}/ifn/ifn4/sig-albacete.zip",                   "IFN4_sig"),  # Albacete
    (f"{_INV}/Sig_03_Alicante.zip",                         "IFN4_sig"),  # Alicante
    (f"{_INV}/sig_asturias_tcm30-536624.zip",               "IFN4_sig"),  # Asturias
    (f"{_INV}/sig_avila_tcm30-536625.zip",                  "IFN4_sig"),  # Ávila
    (f"{_INV}/sig_baleares_tcm30-536626.zip",               "IFN4_sig"),  # Baleares
    (f"{_INV}/sig_burgos_tcm30-536627.zip",                 "IFN4_sig"),  # Burgos
    (f"{_INV}/sig_C%C3%A1diz.accdb.zip",                    "IFN4_sig"),  # Cádiz
    (f"{_INV}/sig_canarias_tcm30-536628.zip",               "IFN4_sig"),  # Canarias
    (f"{_INV}/sig_cantabria_tcm30-536604.zip",              "IFN4_sig"),  # Cantabria
    (f"{_INV}/Sig_12_Castell%C3%B3n.zip",                   "IFN4_sig"),  # Castellón
    (f"{_INV}/sig_cataluna_tcm30-536605.zip",               "IFN4_sig"),  # Cataluña
    (f"{_INV}/ifn/ifn4/sig-ciudadreal.zip",                 "IFN4_sig"),  # Ciudad Real
    (f"{_INV}/sig_C%C3%B3rdoba.accdb.zip",                  "IFN4_sig"),  # Córdoba
    (f"{_INV}/ifn/ifn4/sig-cuenca.zip",                     "IFN4_sig"),  # Cuenca
    (f"{_INV}/sig_extremadura_tcm30-536606.zip",            "IFN4_sig"),  # Extremadura
    (f"{_INV}/ifn/ifn4/sig-guadalajara.zip",                "IFN4_sig"),  # Guadalajara
    (f"{_INV}/Sig_22_Huesca.zip",                           "IFN4_sig"),  # Huesca
    (f"{_INV}/sig_Ja%C3%A9n.accdb.zip",                     "IFN4_sig"),  # Jaén
    (f"{_INV}/sig_leon_tcm30-536608.zip",                   "IFN4_sig"),  # León
    (f"{_INV}/sig_larioja_tcm30-536607.zip",                "IFN4_sig"),  # La Rioja
    (f"{_INV}/sig_lugo_tcm30-536609.zip",                   "IFN4_sig"),  # Lugo
    (f"{_INV}/sig_madrid_tcm30-536610.zip",                 "IFN4_sig"),  # Madrid
    (f"{_INV}/sig_M%C3%A1laga.accdb.zip",                   "IFN4_sig"),  # Málaga
    (f"{_INV}/sig_murcia_tcm30-536611.zip",                 "IFN4_sig"),  # Murcia
    (f"{_INV}/sig_navarra_tcm30-536612.zip",                "IFN4_sig"),  # Navarra
    (f"{_INV}/sig_orense_tcm30-536613.zip",                 "IFN4_sig"),  # Ourense
    (f"{_INV}/sig_paisvasco_tcm30-536614.zip",              "IFN4_sig"),  # País Vasco
    (f"{_INV}/sig_palencia_tcm30-536615.zip",               "IFN4_sig"),  # Palencia
    (f"{_INV}/sig_pontevedra_tcm30-536616.zip",             "IFN4_sig"),  # Pontevedra
    (f"{_INV}/sig_salamanca_tcm30-536617.zip",              "IFN4_sig"),  # Salamanca
    (f"{_INV}/sig_segovia_tcm30-536618.zip",                "IFN4_sig"),  # Segovia
    (f"{_INV}/sig_soria_tcm30-536619.zip",                  "IFN4_sig"),  # Soria
    (f"{_INV}/Sig_44_Teruel.zip",                           "IFN4_sig"),  # Teruel
    (f"{_INV}/ifn/ifn4/sig-toledo.zip",                     "IFN4_sig"),  # Toledo
    (f"{_INV}/Sig_46_Valencia.zip",                         "IFN4_sig"),  # Valencia
    (f"{_INV}/sig_valladolid_tcm30-536620.zip",             "IFN4_sig"),  # Valladolid
    (f"{_INV}/sig_zamora_tcm30-536621.zip",                 "IFN4_sig"),  # Zamora
    (f"{_INV}/Sig_50_Zaragoza.zip",                         "IFN4_sig"),  # Zaragoza
]

# ──────────────────────────────────────────────────────────────────────────────
# Lógica de descarga
# ──────────────────────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    """Sesión con reintentos automáticos ante errores de red."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (IFN bulk downloader)"})
    return session


def filename_from_url(url: str) -> str:
    """Extrae el nombre de fichero desde la URL (último segmento, decodificado)."""
    from urllib.parse import unquote
    return unquote(url.split("/")[-1])


def download_file(url: str, dest_dir: Path, session: requests.Session) -> tuple[str, bool, str]:
    """
    Descarga un fichero si no existe ya.
    Devuelve (filename, ok, mensaje).
    """
    filename = filename_from_url(url)
    dest_path = dest_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 0:
        return filename, True, "ya existe, omitido"

    try:
        with session.get(url, stream=True, timeout=120) as r:
            if r.status_code == 404:
                return filename, False, f"HTTP 404 — URL no encontrada: {url}"
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            tmp_path = dest_path.with_suffix(".tmp")
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    f.write(chunk)
                    downloaded += len(chunk)

            tmp_path.rename(dest_path)
            size_mb = dest_path.stat().st_size / 1024 / 1024
            return filename, True, f"OK ({size_mb:.1f} MB)"

    except Exception as e:
        if dest_path.with_suffix(".tmp").exists():
            dest_path.with_suffix(".tmp").unlink(missing_ok=True)
        return filename, False, f"ERROR: {e}"


def run_downloads(tasks: list[tuple[str, str]], dest_root: Path, workers: int):
    """
    Descarga en paralelo una lista de (url, subcarpeta).
    Muestra progreso y resumen final.
    """
    total = len(tasks)
    ok_count = 0
    err_count = 0
    skipped_count = 0
    errors = []

    session = make_session()

    def do(item):
        url, subdir = item
        return download_file(url, dest_root / subdir, session)

    iterator = range(total)
    if HAS_TQDM:
        bar = tqdm(total=total, unit="fichero", ncols=80)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(do, task): task for task in tasks}
        for future in as_completed(futures):
            filename, ok, msg = future.result()
            if ok:
                if "omitido" in msg:
                    skipped_count += 1
                    log.debug("  %-45s  %s", filename, msg)
                else:
                    ok_count += 1
                    log.info("  ✓ %-43s  %s", filename, msg)
            else:
                err_count += 1
                errors.append((filename, msg))
                log.warning("  ✗ %-43s  %s", filename, msg)

            if HAS_TQDM:
                bar.update(1)
                bar.set_postfix(ok=ok_count, err=err_count, skip=skipped_count)

    if HAS_TQDM:
        bar.close()

    print()
    print(f"{'─'*60}")
    print(f"  Total   : {total}")
    print(f"  Descargados : {ok_count}")
    print(f"  Omitidos    : {skipped_count}  (ya existían)")
    print(f"  Errores     : {err_count}")
    if errors:
        print()
        print("  Ficheros con error:")
        for fname, msg in errors:
            print(f"    · {fname}: {msg}")
    print(f"{'─'*60}")

    return err_count == 0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

ALL_INVENTARIOS = {
    "ifn2":       IFN2_URLS,
    "ifn3_campo": IFN3_CAMPO_URLS,
    "ifn3_sig":   IFN3_SIG_URLS,
    "ifn4_campo": IFN4_CAMPO_URLS,
    "ifn4_sig":   IFN4_SIG_URLS,
}

INVENTARIO_GROUPS = {
    "ifn2": ["ifn2"],
    "ifn3": ["ifn3_campo", "ifn3_sig"],
    "ifn4": ["ifn4_campo", "ifn4_sig"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Descarga masiva de datos IFN2, IFN3 e IFN4 del MITECO"
    )
    parser.add_argument(
        "--dest",
        default="IFN_España",
        help="Carpeta raíz de destino (por defecto: IFN_España)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Número de descargas en paralelo (por defecto: 4)",
    )
    parser.add_argument(
        "--inventarios",
        nargs="+",
        choices=["ifn2", "ifn3", "ifn4"],
        default=["ifn2", "ifn3", "ifn4"],
        help="Qué inventarios descargar (por defecto: los tres)",
    )
    args = parser.parse_args()

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    # Construir lista de tareas
    tasks = []
    for inv in args.inventarios:
        for key in INVENTARIO_GROUPS[inv]:
            tasks.extend(ALL_INVENTARIOS[key])

    print(f"\n  Destino  : {dest_root.resolve()}")
    print(f"  Workers  : {args.workers}")
    print(f"  Ficheros : {len(tasks)}")
    print(f"  Inventarios: {', '.join(args.inventarios)}")
    print()

    grupos = {}
    for url, subdir in tasks:
        grupos.setdefault(subdir, 0)
        grupos[subdir] += 1
    for subdir, n in sorted(grupos.items()):
        print(f"    {subdir:<20} {n:>3} ficheros  →  {dest_root / subdir}")
    print()

    t0 = time.time()
    ok = run_downloads(tasks, dest_root, args.workers)
    elapsed = time.time() - t0
    print(f"  Tiempo total: {elapsed:.0f}s\n")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
