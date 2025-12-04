# Création d'un exécutable autonome

Ce projet peut être empaqueté avec [PyInstaller](https://pyinstaller.org/) pour fournir un binaire qui demande simplement la clé Firecrawl à l'utilisateur.

## 1. Pré-requis
- Python 3.10+
- Dépendances du projet installées (`pip install -r requirements.txt`)
- PyInstaller

```bash
pip install pyinstaller
```

## 2. Lancer l'application via le script dédié

Le script `run_app.py` gère la clé Firecrawl :
- Il lit `FIRECRAWL_API_KEY` depuis les variables d'environnement.
- Sinon, il cherche la clé dans `~/.moto_leboncoin/firecrawl_key.json`.
- À défaut, il demande la clé à l'utilisateur, puis la stocke dans ce fichier pour les prochaines exécutions.

Pour tester sans générer d'exécutable :

```bash
python run_app.py
```

## 3. Générer l'exécutable

Depuis la racine du projet :

```bash
pyinstaller \
  --onefile \
  --name moto-analyzer \
  --collect-all scraper \
  --add-data "data:data" \
  run_app.py
```

- `--onefile` crée un seul binaire (dans `dist/moto-analyzer`).
- `--collect-all scraper` inclut automatiquement les modules `scraper/`.
- `--add-data "data:data"` embarque le dossier `data` (cache, ressources). Ajustez selon vos besoins.

## 4. Utilisation de l'exécutable

1. Copiez `dist/moto-analyzer` sur la machine cible.
2. Au premier lancement, l'utilisateur renseigne sa clé Firecrawl. Elle sera mémorisée dans `~/.moto_leboncoin/firecrawl_key.json`.
3. Pour réinitialiser la clé : `./moto-analyzer --reset-key`.

L'exécutable démarre alors l'application Dash sur `http://localhost:8050/` comme la version Python.
