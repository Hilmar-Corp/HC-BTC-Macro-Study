# Contribution

Toute modification doit préserver la distinction entre refactorisation technique et modification scientifique.

Avant toute contribution :

```bash
uv sync --locked --all-extras
make due-diligence
```

Une refactorisation ne doit pas modifier silencieusement les résultats empiriques gelés.

Toute modification d'un coefficient, d'une statistique, d'une frontière d'échantillon, d'une transformation, d'une date de rupture, d'un résultat de correction de multiplicité ou d'une empreinte scientifique doit être explicitement déclarée comme une nouvelle révision de recherche.

Les données brutes tierces soumises à des restrictions de redistribution ne doivent pas être publiées dans Git.
