# Révision contrôlée du baseline scientifique — 8 septembre 2026

Cette révision du manifeste de reproductibilité est intentionnelle et limitée à deux artefacts.

## Motif

1. La segmentation structurelle est désormais décrite conformément à son implémentation réelle : `custom_segmented_ols_bic`, et non « Bai-Perron ».
2. Le protocole de publication limite la recherche BIC à trois ruptures maximales.
3. Le diagnostic VIF est désormais calculé avec une constante, conformément à la spécification des régressions principales.

## Invariants empiriques

- Le gel numérique principal est inchangé et a été vérifié avant l'approbation de ce manifeste.
- Le BIC sélectionne toujours 0 rupture.
- Les coefficients OLS, covariances HAC, tests de Wald, R², décomposition de Shapley, diagnostics SupF-like et résultats événementiels ne sont pas modifiés par cette révision.

## Artefacts modifiés

### `outputs/tables/structural_break_bic_models.csv`

- ancien SHA-256 : `7d77da05beac58d0c6a69c92ecc2710cf15b7846eb6d7f5209ede2e9194e44ba`
- nouveau SHA-256 : `b3cac91a2a623939e83841c082c6644a4a71683355de30da9eb385f73999fd55`
- ancienne taille : 317 octets
- nouvelle taille : 267 octets

### `outputs/tables/vif.csv`

- ancien SHA-256 : `63b397f16348021978113e109b7883dc5a481117687bc5d4764e28ec0fc5e368`
- nouveau SHA-256 : `5abec9522dedb58235c62f2abb443826b19dfa5573038f3867ed173881863533`
- ancienne taille : 438 octets
- nouvelle taille : 439 octets

## Diagnostic VIF après correction

```text
        model                      factor      vif
primary_ffill              ndx_log_return 1.113260
primary_ffill           dollar_log_return 1.139440
primary_ffill            real_rate_change 1.178679
primary_ffill        credit_spread_change 1.169781
 strict_exact              ndx_log_return 1.117170
 strict_exact    dollar_log_return_strict 1.141768
 strict_exact     real_rate_change_strict 1.179410
 strict_exact credit_spread_change_strict 1.170579
```

Cette révision constitue une modification explicite du contrat d'artefacts, pas une altération silencieuse du baseline.
