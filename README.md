# HC-BTC-Macro-Study

[![CI](https://github.com/Hilmar-Corp/HC-BTC-Macro-Study/actions/workflows/ci.yml/badge.svg)](https://github.com/Hilmar-Corp/HC-BTC-Macro-Study/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12.8-3776AB)
![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue)
![Tests](https://img.shields.io/badge/tests-48%20valid%C3%A9s-brightgreen)
![Pyright](https://img.shields.io/badge/Pyright-0%20erreur-brightgreen)
![Reproductibilité](https://img.shields.io/badge/reproductibilit%C3%A9-environnement%20vierge%20valid%C3%A9-brightgreen)

## Bitcoin est-il devenu un actif macro ?

Ce dépôt contient l'étude quantitative de HilmarCorp consacrée à l'évolution de l'intégration macrofinancière de Bitcoin depuis 2017.

La question étudiée est la suivante :

> **La dépendance de Bitcoin aux facteurs macrofinanciers a-t-elle augmenté depuis 2017 ?**

L'objectif n'est pas de construire un modèle prédictif ni d'attribuer un mécanisme causal unique. L'étude mesure l'évolution, dans le temps, de la relation contemporaine entre les rendements de Bitcoin et plusieurs facteurs de marché : actions américaines, dollar, taux réels et crédit.

---

## Résultat central

Le pouvoir explicatif du modèle macrofinancier est pratiquement nul avant 2020 puis augmente nettement dans les périodes suivantes.

| Période | R² ajusté |
|---|---:|
| Avant 2020 | -0,48 % |
| Mars 2020 – 10 janvier 2024 | 16,70 % |
| Depuis le 11 janvier 2024 | 17,64 % |

La sensibilité estimée au Nasdaq-100 passe approximativement de **0,13** avant 2020 à **0,82** sur la période intermédiaire puis **0,88** depuis janvier 2024.

Les tests conjoints indiquent une modification significative des sensibilités entre la période pré-2020 et les deux périodes suivantes. En revanche, aucune nouvelle rupture conjointe n'est détectée entre la période 2020–2024 et la période postérieure au lancement des ETF spot américains.

L'interprétation retenue est donc :

> **Bitcoin apparaît davantage intégré au risque macrofinancier dans sa synchronisation quotidienne, sans que cela implique une capacité prédictive robuste des facteurs macro sur ses rendements futurs.**

Cette étude ne conclut pas que les ETF spot ont causé cette transformation.

---

## Périmètre empirique

- Début de l'échantillon : **17 août 2017**
- Dernière séance traditionnelle incluse : **4 septembre 2026**
- Panel de marché : **2 275 séances**
- Observations complètes du modèle principal : **2 274**
- Correspondance exacte des clôtures Bitcoin : **100 %**
- Empreinte SHA-256 du panel principal :

```text
2f77822b3e7f599db56026ba6a94d0794de13e2504ee019946ac9b2bc4a4619e
```

Facteurs principaux :

- Nasdaq-100 ;
- dollar américain pondéré par le commerce ;
- taux réel américain à 10 ans ;
- écart de crédit BAA – Treasury.

Des variantes utilisent également le S&P 500 et le VIX.

---

## Contrôles d'intégrité et de reproductibilité

Le dépôt distingue explicitement les résultats scientifiques des contrôles d'ingénierie.

| Contrôle | État |
|---|---|
| Version Python figée | **3.12.8** |
| Dépendances verrouillées | **PASS — `uv.lock`** |
| Fichiers bruts vérifiés par SHA-256 | **PASS — 7 fichiers** |
| Reconstruction du panel depuis les données brutes | **PASS** |
| Vérification du panel | **PASS** |
| Analyse statique Ruff | **PASS** |
| Vérification Pyright | **PASS — 0 erreur** |
| Suite de tests | **PASS — 48 tests** |
| Artefacts scientifiques canoniques | **50** |
| Reproductibilité numérique du gel final | **PASS** |
| Reconstruction en environnement vierge | **PASS** |
| Absence de dépendance à l'historique du répertoire | **PASS** |

Ces contrôles signifient que le dépôt peut reconstruire le corpus scientifique certifié à partir des données brutes autorisées, de l'environnement verrouillé et du code versionné.

Ils ne constituent ni une garantie de performance financière, ni une certification réglementaire, ni une assurance contre le risque de modèle.

---

## Reproductibilité

### Installation exacte

```bash
uv sync --locked --all-extras
```

La version de Python est définie dans :

```text
.python-version
```

Les dépendances exactes sont définies dans :

```text
uv.lock
```

Une exportation lisible est également disponible dans :

```text
requirements.lock.txt
```

### Contrôle de diligence

```bash
make due-diligence
```

Ce contrôle vérifie notamment :

- l'intégrité du verrouillage des dépendances ;
- les empreintes SHA-256 des données brutes ;
- la compilation ;
- Ruff ;
- Pyright ;
- l'architecture du dépôt ;
- les tests ;
- l'identité numérique du résultat gelé.

### Reconstruction complète

Avec le paquet privé de données brutes autorisé :

```bash
make clean-room
```

La procédure :

1. exporte un arbre Git propre depuis le commit courant ;
2. restaure les données brutes certifiées ;
3. recrée l'environnement depuis `uv.lock` ;
4. vérifie les empreintes SHA-256 ;
5. reconstruit l'ensemble du corpus empirique ;
6. exécute la suite de tests sur les résultats reconstruits ;
7. compare le gel numérique obtenu au gel certifié.

---

## Données et traçabilité

La provenance des sources est décrite dans :

```text
data/provenance.yaml
```

Les empreintes cryptographiques des données brutes sont définies dans :

```text
reproducibility/manifests/raw_data_manifest.json
```

Le gel scientifique de référence est défini dans :

```text
reproducibility/baseline/final_results_freeze.json
```

Le contrat canonique des artefacts scientifiques est défini dans :

```text
reproducibility/baseline/artifact_manifest.json
```

Les données brutes provenant de fournisseurs tiers ne sont volontairement pas publiées dans ce dépôt. Leur utilisation et leur redistribution restent soumises aux conditions des fournisseurs concernés.

---

## Architecture

```text
src/hc_macro_integration/
├── cli.py
├── config.py
├── contracts.py
├── integrity.py
├── paths.py
├── pipeline.py
├── reproducibility.py
├── run_manifest.py
├── data/
├── models/
├── robustness/
└── reporting/
```

Les modules sont organisés par responsabilité scientifique et technique, et non par succession historique de « phases ».

---

## Commandes principales

```bash
hc-macro status
hc-macro data
hc-macro core
hc-macro robustness
hc-macro events
hc-macro freeze
hc-macro verify-freeze
hc-macro all
```

`hc-macro all` constitue le chemin de reconstruction figée. Il reconstruit les sorties dérivées à partir des données brutes disponibles et vérifie l'identité du gel scientifique.

Le rafraîchissement des données externes reste une opération explicite via :

```bash
hc-macro data
```

---

## Intégration continue

GitHub Actions contrôle automatiquement sur les changements publics :

- le verrouillage de l'environnement ;
- la compilation ;
- Ruff ;
- Pyright ;
- les tests ne nécessitant pas les données brutes privées ;
- l'audit des dépendances.

Une procédure distincte de reconstruction complète est disponible pour les exécutions autorisées disposant du paquet privé de données.

---

## Limites scientifiques

Les résultats décrivent des associations contemporaines et des changements de structure statistique.

Ils ne permettent pas, à eux seuls, d'établir :

- une causalité ;
- une date causale unique de rupture ;
- une capacité prédictive stable à 1, 5 ou 20 séances ;
- un effet causal des ETF spot ;
- une recommandation d'investissement.

Les tests événementiels sont traités comme des résultats descriptifs et secondaires après correction de multiplicité.

---

## Licence

Le code source original et la documentation de HilmarCorp présents dans ce dépôt sont distribués sous **Apache License 2.0**.

Le fichier [`LICENSE`](LICENSE) contient le texte officiel complet de la licence.

Le fichier [`NOTICE`](NOTICE) précise le périmètre des droits ainsi que le traitement des données provenant de tiers.

La licence Apache 2.0 ne s'applique pas automatiquement aux données de marché ou macroéconomiques provenant de fournisseurs tiers.

---

## Avertissement

Ce dépôt constitue un travail de recherche quantitative.

Il ne constitue pas :

- un conseil en investissement ;
- une recommandation personnalisée ;
- une sollicitation d'achat ou de vente ;
- une promesse de performance ;
- une garantie de rendement.

Les résultats historiques et statistiques ne préjugent pas des résultats futurs.

---

Copyright © 2026 HilmarCorp.
