# Markov-Vision 👁️

**Markov-Vision** est une application de bureau interactive de traitement d'images et de synthèse de texture développée en Python. Elle s'appuie sur des concepts mathématiques de **Champs Aléatoires de Markov (MRF)** et l'algorithme de **l'échantillonneur de Gibbs** pour segmenter des images et générer des textures en temps réel. 

L'application offre une interface graphique fluide, robuste et multithreadée avec une visualisation en direct des cartes de classes et des courbes de convergence (énergie globale).

---

## 🚀 Fonctionnalités Clés

*   **Workflow A : Segmentation d'Images**
    *   Classification de pixels en $K$ classes à l'aide d'un modèle de Potts et d'une cohésion spatiale réglable.
    *   Visualisation dynamique de la convergence à l'aide d'un graphique d'énergie globale en direct.
*   **Workflow B : Synthèse de Texture Réaliste**
    *   Génération d'une texture étendue à partir d'un petit échantillon source (patch).
    *   Maintien de la cohérence visuelle et chromatique par un mapping intelligent des étiquettes vers les couleurs naturelles de l'échantillon.
    *   Algorithme guidé par un terme de données spatiales pour reproduire les motifs géométriques et filons (ex: bois, tissu).
*   **Échantillonneur de Gibbs Vectorisé (Checkerboard)**
    *   Accélération de **35x à plus de 300x** par rapport à la boucle pixel par pixel traditionnelle en parallélisant les mises à jour sur une grille de pixels bipartie (damier).
*   **Contrôle Interactif & Temps Réel**
    *   Modification en direct de la force de cohésion spatiale ($\beta$) et de la température ($T$) via des réglettes (sliders).
    *   **Debouncing** (anti-rebond) de 200 ms sur les sliders pour éviter les surcharges de calculs lors des glissements rapides.
    *   Changement dynamique des plages limites (valeurs maximales) pour $\beta$ et $T$ directement depuis l'interface.
*   **Architecture Robuste**
    *   Exécution asynchrone des calculs lourds dans un thread d'arrière-plan pour maintenir l'interface Tkinter réactive.
    *   Utilisation d'une boucle de scrutation non bloquante (`root.after`) pour éliminer tout risque de crash de mémoire ou de corruption de tas (`alloc: invalid block`) sous Windows.

---

## 📐 Concepts Mathématiques

### 1. Modélisation par Champ de Markov (MRF)
Le réseau de pixels est modélisé comme un graphe non orienté où chaque pixel $i$ est un nœud associé à une variable aléatoire $X_i \in \{0, \dots, K-1\}$ représentant son étiquette de classe. 

### 2. Modèle de Potts
L'énergie locale du pixel $i$ avec l'étiquette candidate $k$ est définie par :
$$E_i(k) = -\beta \sum_{j \in \mathcal{N}(i)} \mathbb{I}(X_j = k)$$
où :
*   $\mathcal{N}(i)$ désigne les voisins directs du pixel $i$ (voisinage de 4-connexité : haut, bas, gauche, droite).
*   $\mathbb{I}$ est la fonction indicatrice (vaut $1$ si le voisin $j$ a la même étiquette $k$, $0$ sinon).
*   $\beta \ge 0$ contrôle la force de cohésion spatiale. Une valeur élevée favorise les régions homogènes.

Dans le **Workflow B**, un terme de données spatiales est ajouté à l'énergie pour aligner la texture synthétisée avec la structure du patch d'origine :
$$E_i(k) = -\beta \sum_{j \in \mathcal{N}(i)} \mathbb{I}(X_j = k) - \alpha \cdot \mathbb{I}(\text{ref}_i = k)$$
où $\alpha$ est le poids accordé au guidage par l'image de référence et $\text{ref}_i$ est l'étiquette de référence.

### 3. Distribution de Gibbs et Échantillonnage
La probabilité de transition pour le pixel $i$ d'adopter l'étiquette $k$ sachant ses voisins est donnée par le filtre de Gibbs :
$$P(X_i = k \mid X_{-i}) = \frac{\exp\left(-\frac{E_i(k)}{T}\right)}{\sum_{k'=0}^{K-1} \exp\left(-\frac{E_i(k')}{T}\right)}$$
où $T > 0$ représente la température de relaxation. Plus $T$ est élevé, plus le système accepte des configurations aléatoires.

### 4. Vectorisation par Coloration Bipartie (Damier)
Dans une grille de pixels 2D en 4-connexité, le graphe est biparti. Cela signifie que l'on peut diviser les pixels en deux sous-ensembles :
1.  **Pixels Pairs** ($i + j$ pair, couleur blanche du damier)
2.  **Pixels Impairs** ($i + j$ impair, couleur noire du damier)

Puisque les pixels pairs ne dépendent que de leurs voisins impairs (et vice-versa), nous pouvons mettre à jour tous les pixels pairs en parallèle à l'aide d'opérations vectorielles NumPy, puis faire de même pour les pixels impairs. Cela évite le parcours séquentiel lent en Python et apporte un gain de performance massif.

---

## 📂 Architecture du Projet

Le projet est structuré de façon modulaire :

```
markov-vision/
│
├── sample_images/          # Images de test (ladybug, textures de bois, etc.)
│   ├── 4.1.05.tiff
│   ├── imageB.jpg
│   ├── imageB1.jpg
│   ├── images.jpg
│   ├── ladybug.jpg
│   └── test.jpg
│
├── tests/                  # Tests unitaires
│   └── test_gibbs.py
│
├── config.py               # Constantes globales et hyperparamètres par défaut
├── initializer.py          # Initialisation des étiquettes (aléatoire, grille, etc.)
├── image_loader.py         # Chargement et redimensionnement d'image avec Pillow
├── gibbs_sampler.py        # Échantillonneurs de Gibbs (standard et vectorisé damier)
├── texture_synth.py        # Logique de synthèse de texture et colorisation naturelle
├── convergence.py          # Calcul de l'énergie globale pour la courbe de convergence
├── main.py                 # Interface graphique Tkinter et orchestration multithread
├── requirements.txt        # Dépendances Python requises
└── .gitignore              # Fichiers exclus de Git
```

---

## 🛠️ Installation & Exécution

### Prérequis
*   Python 3.10 ou version supérieure installée.

### 1. Cloner le dépôt et se placer dans le répertoire
```bash
git clone <url-du-depot-github>
cd markov-vision
```

### 2. Créer et activer un environnement virtuel (Recommandé)
Sur Windows (PowerShell) :
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Sur Linux / macOS :
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Lancer l'application
```bash
python main.py
```

---

## 🧪 Lancement des Tests

Le projet intègre des tests unitaires pour valider le comportement et l'intégrité de l'échantillonneur de Gibbs. Pour exécuter les tests :

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 👥 Auteurs
Développé dans le cadre d'un projet universitaire sur le traitement d'images probabiliste et les modèles de vision de Markov.