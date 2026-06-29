# DP-representations — pipeline reproductible

Comparaison des représentations internes (via **CKA** — Centered Kernel
Alignment) entre un ResNet18 entraîné normalement et un ResNet18 entraîné
avec **DP-SGD** (Opacus) sur CIFAR-10.

Ce dépôt remplace les deux notebooks Colab d'origine (`dpsgd_baseline.ipynb`
et `cka_analysis.ipynb`) par une pipeline de scripts Python, pour pouvoir
relancer des expériences de façon reproductible, sans recopier de cellules
ni risquer un nommage de fichier incohérent entre deux runs.

## Structure du projet

```
dp_representations/
├── configs/
│   ├── default.yaml        # config par défaut (epsilon=10, seed=42, 10 epochs)
│   └── eps_low.yaml         # exemple de variante (epsilon=2)
├── src/
│   ├── model.py             # définition du ResNet18 DP-compatible (architecture partagée)
│   ├── data.py               # téléchargement + chargement CIFAR-10
│   ├── config.py             # dataclass Config + chargement YAML + overrides CLI
│   ├── checkpoint.py         # nommage de fichiers (avec timestamp) + save/load
│   ├── training.py           # boucles d'entraînement baseline et DP-SGD
│   ├── cka.py                 # formules CKA + extraction d'activations
│   ├── visualization.py      # tracés matplotlib (courbes, heatmaps)
│   ├── logging_utils.py      # logging console + fichier
│   ├── train_baseline.py     # script CLI : entraînement sans DP
│   ├── train_dp.py            # script CLI : entraînement DP-SGD
│   ├── run_cka.py              # script CLI : analyse CKA (3 sous-commandes)
│   └── run_pipeline.py        # orchestrateur optionnel (baseline → DP → CKA)
├── networks/                  # checkpoints .pth + métadonnées .json (générés)
├── results/                   # figures et résumés CKA (générés)
├── logs/                       # logs texte par run (générés)
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate         # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

CIFAR-10 est téléchargé automatiquement (depuis le miroir FastAI) au premier
lancement d'un script d'entraînement ou de `run_cka.py`, dans `./cifar10`.

## Ce que fait la pipeline

### 1. Architecture partagée (`model.py`)

Le point critique des notebooks d'origine : pour comparer les
représentations de deux modèles avec CKA, **il faut pouvoir recharger les
deux checkpoints dans une architecture strictement identique**, sinon
`load_state_dict` échoue ou — pire — réussit silencieusement sur une
architecture légèrement différente, et le CKA ne compare plus la même chose.

`model.py` centralise donc la construction du ResNet18 "DP-compatible"
(GroupNorm au lieu de BatchNorm, ReLU non-inplace, résiduel non-inplace) en
un seul endroit, utilisé à la fois par l'entraînement (baseline **et** DP) et
par le chargement pour le CKA. Plus aucun risque de divergence entre les
deux notebooks d'origine.

### 2. Entraînement (`train_baseline.py` / `train_dp.py`)

Deux scripts symétriques :
- `train_baseline.py` : optimisation standard (RMSprop), pas de DP.
- `train_dp.py` : DP-SGD via Opacus. `PrivacyEngine.make_private_with_epsilon`
  calcule automatiquement le bruit (sigma) nécessaire pour atteindre le
  budget `(epsilon, delta)` demandé sur le nombre d'epochs choisi.
  `BatchMemoryManager` découpe le batch logique en petits batchs physiques
  pour tenir en mémoire malgré le calcul de gradients par échantillon.

Chaque script sauvegarde :
- un fichier `.pth` (poids + historique d'accuracy + hyperparamètres),
- un fichier `.json` jumeau (mêmes métadonnées, sans les poids) — pratique
  pour explorer rapidement tous les runs passés sans charger de tenseurs.

### 3. Nommage des checkpoints (`checkpoint.py`)

Convention :
```
{prefix}_eps{epsilon}_delta{delta}_epoch{epochs}_C{max_grad_norm}_seed{seed}_{timestamp}.pth
```
Exemple : `dp_resnet18_eps10_delta1e08_epoch10_C1.2_seed42_20260628-143012.pth`

Le **timestamp** (`YYYYmmdd-HHMMSS`) est ajouté par rapport aux notebooks
d'origine : il garantit qu'on ne réécrase jamais un ancien run par accident,
même en relançant exactement les mêmes hyperparamètres. Comme le format est
triable lexicographiquement, `find_latest_checkpoint()` retrouve facilement
"le dernier baseline avec seed=42" sans avoir à copier-coller un nom de
fichier.

### 4. Analyse CKA (`run_cka.py`)

Trois sous-commandes, qui correspondent aux analyses du notebook 2 :

- **`compare`** : CKA entre deux checkpoints quelconques (baseline vs DP,
  baseline vs baseline avec un autre seed, DP vs DP...) sur les couches
  haut-niveau (`layer1` à `layer4`). C'est la comparaison la plus courante.
- **`multi-eps`** : compare une baseline de référence à plusieurs
  checkpoints DP entraînés avec des epsilons différents, pour visualiser
  comment la similarité de représentation évolue avec le budget de
  confidentialité.
- **`fine-grained`** : la heatmap "en damier" à grain fin (sous-couches
  conv1/conv2 de chaque bloc résiduel), pour visualiser des motifs internes
  plus précis qu'au niveau des 4 stages principaux.

Chaque sous-commande sauvegarde une figure `.png` et les valeurs numériques
brutes (`.json` ou `.npy`) dans `results/`, avec timestamp.

### 5. Orchestrateur (`run_pipeline.py`)

Enchaîne `train_baseline.py` → `train_dp.py` → `run_cka.py compare`
automatiquement, en réutilisant les checkpoints les plus récents. Pratique
pour un run complet "par défaut", mais les scripts séparés restent
recommandés au quotidien (voir plus bas) car on a rarement besoin de tout
relancer à chaque fois.

## Configuration

Toute la config vit dans un fichier YAML (voir `configs/default.yaml`), et
chaque champ peut être surchargé en ligne de commande :

```bash
# Utilise toute la config par défaut
python src/train_dp.py --config configs/default.yaml

# Pareil, mais avec un epsilon différent juste pour cet essai
python src/train_dp.py --config configs/default.yaml --epsilon 2

# Sans fichier config du tout (valeurs par défaut du dataclass Config)
python src/train_baseline.py --epochs 5 --seed 43
```

---

## Workflow quotidien (usage perso)

Ordre habituel pour une session de travail :

### A. Première fois / setup d'un nouvel epsilon à tester

```bash
cd src

# 1. Entraîner la baseline une fois (réutilisable pour tous les epsilons)
python train_baseline.py --config ../configs/default.yaml

# 2. Entraîner le modèle DP pour l'epsilon voulu
python train_dp.py --config ../configs/default.yaml --epsilon 8

# 3. Comparer les deux (trouver les noms exacts avec `ls ../networks/`,
#    ou utiliser find_latest_checkpoint depuis un shell Python si tu préfères)
python run_cka.py compare \
    --ckpt-a ../networks/baseline_resnet18_..._seed42_<timestamp>.pth \
    --ckpt-b ../networks/dp_resnet18_eps8_..._seed42_<timestamp>.pth \
    --label-a Baseline --label-b "DP_eps8"
```

### B. Itérer rapidement sur un seul paramètre

Pas besoin de toucher au YAML pour un essai ponctuel — juste l'override CLI :

```bash
python train_dp.py --config ../configs/default.yaml --max-grad-norm 0.8 --seed 7
```

### C. Tout enchaîner d'un coup (run complet)

```bash
python run_pipeline.py --config ../configs/default.yaml --epsilon 8
```

Ajoute `--skip-baseline` si tu as déjà une baseline pour ce seed et ne veux
pas la réentraîner.

### D. Balayage epsilon (graphique de synthèse)

Après avoir entraîné plusieurs modèles DP (epsilon=2, 8, 10 par exemple) :

```bash
python run_cka.py multi-eps \
    --baseline-ckpt ../networks/baseline_resnet18_..._seed42_<ts>.pth \
    --dp-ckpts ../networks/dp_resnet18_eps2_..._<ts>.pth \
               ../networks/dp_resnet18_eps8_..._<ts>.pth \
               ../networks/dp_resnet18_eps10_..._<ts>.pth \
    --epsilons 2 8 10
```

### E. Retrouver un ancien run

Toutes les métadonnées (sans les poids) sont dans les `.json` à côté de
chaque `.pth` dans `networks/`. Pour parcourir rapidement :

```bash
cat ../networks/dp_resnet18_eps8_*.json | python -m json.tool
```

ou en Python :

```python
import json
from pathlib import Path
for p in sorted(Path("networks").glob("*.json")):
    meta = json.load(open(p))
    print(p.name, "->", meta.get("test_acc_history", [None])[-1])
```

### F. Garder un dossier propre

- `networks/` peut grossir vite (chaque run garde son `.pth`, jamais
  écrasé grâce au timestamp). Pense à faire un ménage périodique des runs
  d'essai que tu ne veux pas garder.
- Les logs texte complets (`logs/*.log`) contiennent tout l'historique
  epoch par epoch — utile en cas de plantage à mi-parcours pour savoir où ça
  s'est arrêté.

## Notes de reproductibilité

- `set_seed()` fixe `random`, `numpy` et `torch` (CPU + CUDA). Le seed est
  aussi stocké dans chaque checkpoint.
- La normalisation CIFAR-10 (moyenne/écart-type) est définie une seule fois
  dans `data.py` et utilisée identiquement par l'entraînement et le CKA :
  toute divergence ici invaliderait la comparaison de représentations.
- Aucune augmentation de données n'est appliquée, pour éviter d'introduire
  une variance supplémentaire qui brouillerait la comparaison baseline/DP.
