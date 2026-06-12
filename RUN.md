Installation et exécution rapide

1) Créer un environnement virtuel (Windows PowerShell):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

2) Installer les dépendances:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

3) Lancer l'application:

    python main.py

Remarques:
- Si vous utilisez CMD, activez le venv avec `.\.venv\\Scripts\\activate.bat`.
- Sous Linux/macOS, adaptez la commande d'activation.
