# Avant de remplacer ton projet

Cette archive est une **version adaptée à vérifier**, pas une certification de
réussite à la soutenance. Elle contient les sources, le SDK Python joint conservé
à l’identique, un packaging SDK reconstruit, un verrou uv, des tests et un audit.

1. Décompresse dans un **nouveau dossier** et conserve ton projet actuel intact.
2. Lis `AUDIT.md` puis `VALIDATION.txt` : ils distinguent ce qui a été vérifié de
   ce qui nécessite le vrai modèle.
3. Remplace `<your_login>` sur la première ligne du README anglais.
4. Depuis la racine extraite : `uv sync`, puis `make lint` et `make test`.
5. Lance `uv run python -m src`, puis vérifie les appels produits dans
   `data/output/function_calling_results.json`.
6. Mesure le temps sur ta machine et compare **tous** les arguments aux attentes,
   pas seulement le nom de fonction. Les tests unitaires ne mesurent pas le LLM.
7. Fais relire les changements et assure-toi de pouvoir expliquer chaque état
   de la grammaire avant de les rendre.

Ton arborescence indiquait un SDK et des fichiers de verrouillage dont certains
n’étaient pas téléversés. Ils ne sont donc pas prétendument reproduits ici :
`llm_sdk/pyproject.toml` est reconstruit, `uv.lock` est régénéré et le Makefile est
nouveau. Le code de `init.py` devient `llm_sdk/llm_sdk/__init__.py` sans changement.
L’ancien `main.py` correspond ici à `src/__main__.py`, conformément à l’entrée
`python -m src` ; ne conserve pas une seconde implémentation concurrente.

Le modèle et ses poids ne sont pas inclus. L’installation complète et l’inférence
Qwen n’ont pas pu être exécutées dans l’espace disque disponible. Le verrou a été
vérifié et une installation partielle des dépendances de développement a été
réalisée ; cela ne vaut pas un `uv sync` complet validé.
