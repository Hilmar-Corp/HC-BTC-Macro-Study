# Figures de publication — Bitcoin est-il devenu un actif macro ?

Les figures de publication sont générées séparément des figures exploratoires du pipeline. Elles reprennent la grammaire visuelle des notes de recherche HilmarCorp : fond blanc, hiérarchie noir / gris / bleu HC, absence d'éléments décoratifs et priorité à l'information empirique.

## Génération

Après reconstruction du corpus empirique :

```bash
make publication-figures
```

Les fichiers sont écrits dans :

```text
outputs/figures/publication/
```

## Contrat graphique

1. `01_synchronisation_temporelle.png` — alignement exact de Bitcoin sur les clôtures effectives Nasdaq.
2. `02_r2_ajuste_par_periode.png` — R² ajusté du modèle macrofinancier par période.
3. `03_r2_roulant_252.png` — R² ajusté roulant sur 252 séances.
4. `04_beta_nasdaq_par_periode.png` — sensibilité contemporaine au Nasdaq-100 par période.
5. `05_betas_roulants_252.png` — quatre sensibilités roulantes avec bandes ponctuelles HAC à 95 %.
6. `06_shapley_r2_absolu.png` — contributions absolues de Shapley au R².
7. `07_nasdaq_vs_sp500.png` — robustesse Nasdaq-100 contre S&P 500.
8. `08_profils_rupture.png` — profils exacts de la statistique de changement selon la date candidate.
9. `09_evenements_meme_jour.png` — réponses contemporaines de Bitcoin aux principaux événements extrêmes.
10. `10_evenements_horizons.png` — différences P3-P1 par événement et horizon, avec q-values Benjamini-Hochberg.

Les graphiques de rupture sont reconstruits à partir des données déjà présentes dans le corpus local. La génération de publication ne télécharge pas silencieusement de nouvelles données de marché.

Les figures de l'étude événementielle conservent explicitement le résultat négatif après correction de multiplicité : aucune cellule n'est marquée comme robuste lorsque `reject_fdr_5pct` est faux.

