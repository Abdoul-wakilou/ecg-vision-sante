"""Télécharge les datasets publics du projet depuis PhysioNet.

Utilisation :
    python scripts/download_data.py mitdb          # MIT-BIH, 77 Mo (exercice 1)
    python scripts/download_data.py ptbxl          # PTB-XL, ZIP de 1,8 Go (exercices 2 à 5)
    python scripts/download_data.py ptbxl --full   # garde aussi les signaux à 500 Hz
    python scripts/download_data.py all

Le script n'utilise que la bibliothèque standard de Python : il marche avant
même d'installer requirements.txt. Un téléchargement interrompu reprend là où
il s'était arrêté si on relance la même commande.

Les données arrivent dans data/<nom>/ (dossier exclu de Git).
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DATASETS = {
    "mitdb": {
        "nom": "MIT-BIH Arrhythmia Database v1.0.0",
        "url": "https://physionet.org/content/mitdb/get-zip/1.0.0/",
        "licence": "Open Data Commons Attribution License v1.0",
        "page": "https://physionet.org/content/mitdb/1.0.0/",
    },
    "ptbxl": {
        "nom": "PTB-XL v1.0.3",
        "url": "https://physionet.org/content/ptb-xl/get-zip/1.0.3/",
        "licence": "Creative Commons Attribution 4.0 (CC BY 4.0)",
        "page": "https://physionet.org/content/ptb-xl/1.0.3/",
    },
}


def telecharger(url: str, destination: Path) -> None:
    """Télécharge url vers destination, en reprenant un fichier partiel s'il existe."""
    partiel = destination.with_suffix(destination.suffix + ".part")
    deja = partiel.stat().st_size if partiel.exists() else 0

    requete = urllib.request.Request(url, headers={"User-Agent": "ecg-vision-sante"})
    if deja:
        requete.add_header("Range", f"bytes={deja}-")

    with urllib.request.urlopen(requete) as reponse:
        reprise = reponse.status == 206
        if deja and not reprise:
            deja = 0  # le serveur ignore la reprise : on recommence
        total = deja + int(reponse.headers.get("Content-Length", 0))
        mode = "ab" if reprise else "wb"
        with open(partiel, mode) as fichier:
            recu = deja
            while bloc := reponse.read(1024 * 1024):
                fichier.write(bloc)
                recu += len(bloc)
                if total:
                    print(f"\r  {recu / 1e6:8.1f} / {total / 1e6:.1f} Mo "
                          f"({100 * recu / total:5.1f} %)", end="", flush=True)
    print()
    partiel.replace(destination)


def extraire(archive: Path, cible: Path, garder_500hz: bool) -> int:
    """Extrait l'archive dans cible, sans le dossier racine du ZIP.

    Pour PTB-XL, ignore records500/ sauf si garder_500hz est vrai.
    Renvoie le nombre de fichiers extraits.
    """
    n = 0
    with zipfile.ZipFile(archive) as zf:
        for membre in zf.infolist():
            if membre.is_dir():
                continue
            # "ptb-xl-.../records100/00000/00001_lr.hea" -> "records100/00000/00001_lr.hea"
            relatif = Path(*Path(membre.filename).parts[1:])
            if not relatif.parts:
                continue
            if relatif.parts[0] == "records500" and not garder_500hz:
                continue
            chemin = cible / relatif
            chemin.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(membre) as src, open(chemin, "wb") as dst:
                dst.write(src.read())
            n += 1
    return n


def installer(cle: str, garder_500hz: bool = False, garder_zip: bool = False) -> None:
    info = DATASETS[cle]
    cible = DATA_DIR / cle
    archive = DATA_DIR / f"{cle}.zip"

    print(f"\n=== {info['nom']} ===")
    if (cible / "RECORDS").exists():
        print(f"  Déjà présent dans {cible.relative_to(ROOT)}, rien à faire.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    if not archive.exists():
        print(f"  Téléchargement depuis {info['url']}")
        telecharger(info["url"], archive)

    print("  Extraction...")
    n = extraire(archive, cible, garder_500hz)
    # RECORDS est la liste officielle des enregistrements fournie par PhysioNet
    nb_records = len((cible / "RECORDS").read_text().split())
    print(f"  {n} fichiers extraits, {nb_records} enregistrements listés dans RECORDS.")

    if not garder_zip:
        archive.unlink()

    print(f"  Licence : {info['licence']}. À citer : {info['page']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=[*DATASETS, "all"])
    parser.add_argument("--full", action="store_true",
                        help="PTB-XL : garder aussi records500/ (signaux à 500 Hz, ~3 Go de plus)")
    parser.add_argument("--keep-zip", action="store_true",
                        help="ne pas supprimer l'archive ZIP après extraction")
    args = parser.parse_args()

    cles = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for cle in cles:
        installer(cle, garder_500hz=args.full, garder_zip=args.keep_zip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
