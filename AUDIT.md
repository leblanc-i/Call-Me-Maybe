# Audit du projet Call Me Maybe

## Périmètre et verdict

L’architecture initiale sépare déjà le parser, les prompts, le modèle et le
décodage. Le principal écart ne réside pas dans les dossiers : il concerne le
traitement des tokens et les sorties de secours du décodeur. [file:6][file:8][file:10][file:11]

Le sujet demande une génération contrainte par le schéma, une fonction choisie
par le LLM, une CLI `python -m src`, des validations et une gestion propre des
erreurs. Une conformité totale ne peut pas être attestée sans mesurer aussi
l’extraction des arguments et les performances avec le vrai modèle. [file:12]

La pièce jointe `init.py` a été interprétée comme l’implémentation du SDK ;
`main.py` comme l’entrée de l’application. Les manifestes SDK, les fichiers de
verrouillage, le Makefile original et les `__init__.py` de tous les paquets
n’étaient pas joints. L’arborescence fournie établit leur existence, pas leur
contenu. Aucun diagnostic de leur contenu absent n’est donc affirmé.

## Écarts du code initial

| Priorité | Observation vérifiable | Conséquence | Modification livrée |
|---|---|---|---|
| Bloquant | Le nombre est retourné dès que le premier préfixe est convertible. [file:6] | Un nombre réparti sur plusieurs tokens peut être tronqué. | Automate JSON numérique ; arrêt uniquement au délimiteur suivant. |
| Bloquant | Les chaînes du vocabulaire sont concaténées sans décodage byte-level. [file:6] | Confusion entre représentation BPE et texte produit. | Inversion de l’alphabet byte-level et reconstruction UTF-8 incrémentale. |
| Bloquant | Un échec de choix de fonction se termine par le plus long préfixe commun ou la première fonction. [file:6] | La décision finale peut ne plus provenir d’un choix valide du modèle. | Aucun fallback heuristique ; erreur explicite si la génération échoue. |
| Élevé | Les conversions finales peuvent substituer `0` ou un booléen par défaut. [file:6] | Des arguments inventés deviennent des appels apparemment réussis. | Aucune valeur de remplacement ; refus de l’appel incomplet. |
| Élevé | Le test de préfixe d’un booléen accepte aussi des prolongements d’une valeur déjà complète. [file:6] | Grammaire trop permissive pour des littéraux exacts. | Choix strict entre `true`, `false` et fin légale du littéral. |
| Élevé | Les strings ne sont contraintes que par l’absence de saut de ligne, puis nettoyées avec `strip`. [file:6] | Pas de vraie grammaire de chaîne JSON ; espaces utiles potentiellement perdus. | Guillemets, échappements et contrôles gérés dans la grammaire ; aucun nettoyage du contenu. |
| Élevé | `LLM` et `Parser` sont des classes ordinaires, contrairement à la consigne sur Pydantic. [file:8][file:10][file:12] | Écart de validation des classes applicatives. | Services et états applicatifs fondés sur BaseModel ; exception conservée comme exception. |
| Élevé | `main.py` ne réalise pas la vérification finale complète annoncée dans le README. [file:7][file:5] | Des clés ou types incorrects peuvent atteindre la construction de sortie. | Validation indépendante des clés exactes, du nom et des types sans coercition. |
| Moyen | Le vocabulaire est relu pour chaque appel. [file:6] | Travail répété inutile sur un même catalogue de tokens. | Chargement unique et trie réutilisé. |
| Moyen | Le CPU est forcé dans le wrapper. [file:8] | Accélération automatique du SDK ignorée. | Sélection automatique du SDK ou `--device` explicite ; distribution Linux CPU documentée. |
| Moyen | Le README conserve `<your_login>` et ses ressources sont peu précises. [file:5] | Documentation pas encore personnalisée pour la remise. | README anglais détaillé, liens réels ; login laissé explicitement à confirmer. |

Le `pyproject.toml` initial déclare directement les bibliothèques internes du
SDK. Cela ne suffit pas à prouver un import interdit dans le code applicatif,
mais ne distingue pas correctement les responsabilités de packaging ; la
version adaptée les déclare dans le paquet SDK local uniquement. [file:4][file:3]

## Décodage adapté

L’ancienne stratégie assemble elle-même l’enveloppe JSON après plusieurs petites
générations. Ce découpage n’est pas, à lui seul, une preuve de non-conformité :
le problème vérifiable est la faiblesse des contraintes et des terminaisons.
La variante livrée contraint l’appel complet pour rendre l’invariant du sujet
plus directement démontrable. [file:6][file:12]

Les noms de fonctions et de paramètres proviennent exclusivement du fichier de
définition. Les logits choisissent les tokens permis ; aucun mapping entre mots
du prompt et noms de fonctions n’est codé à la main. Les noms qui partagent un
préfixe sont distingués par leur chaîne JSON complète, guillemet final compris.

Le masque invalide les tokens complets, et non seulement leur premier caractère.
Un token peut franchir une frontière entre valeur et ponctuation ; un caractère
UTF-8 peut aussi être partagé entre plusieurs tokens. Le vocabulaire est
reconstruit en bytes plutôt qu’en utilisant `decode([id])` token par token.

Le prompt utilise le cadrage Qwen3 et un bloc de pensée déjà fermé. Cette forme
est dérivée du template public de Qwen3 ; elle ne nécessite ni accès au tokenizer
privé du SDK ni import de Transformers dans `src`. [web:50][web:33]

## Contrat des données

Les données livrées utilisent des types scalaires ; la solution prend en charge
`number`, `string`, `boolean`, plus `integer` et `null`. Les objets et tableaux
imbriqués ne sont pas implémentés : ils restent hors de cette adaptation et
sont refusés proprement, sans prétendre implémenter tout JSON Schema.

Un nombre peut contenir un signe négatif, une fraction ou un exposant. Une
validation finale refuse les flottants non finis et les chaînes Unicode non
encodables. Un booléen n’est pas accepté comme nombre malgré le sous-typage
Python de `bool` par rapport à `int`.

Les modèles utilisent la validation stricte et interdisent les champs
supplémentaires. Les objets JSON à clés répétées sont refusés dès la lecture,
au lieu de laisser le parseur écraser silencieusement une valeur. Le choix de
BaseModel et la séparation des attributs privés suivent les mécanismes publics
de Pydantic. [web:51][web:36]

## Installation et intégrité

Le code fourni du SDK est conservé octet pour octet ; son hash est inscrit dans
`VALIDATION.txt`. Le manifeste du SDK et sa réexportation publique sont
reconstruits et identifiés comme tels. Il n’y a pas d’accès à `_model`,
`_tokenizer` ou `_device` d’un objet provenant du SDK dans `src`.

Le workspace uv fait du SDK un paquet local et permet au réglage d’index CPU de
s’appliquer à ses dépendances. Le verrou a été régénéré pour ces manifestes : il
ne s’agit pas de prétendre avoir analysé le verrou original absent. Ce montage
s’appuie sur le mécanisme documenté des workspaces. [web:55]

`numpy` est limité à une version compatible avec le contrôle de typage visant
Python 3.10 ; les versions de Transformers et de PyTorch sont déclarées dans
le SDK reconstruit. Ces contraintes constituent un choix de packaging, pas une
modification des appels internes du SDK officiel.

## Sortie et erreurs

Le chemin par défaut devient
`data/output/function_calling_results.json`, selon V.4. Le sujet contient aussi
un exemple avec `function_calls.json` et un autre raccourci de chemin ;
`--output` permet de choisir explicitement un autre emplacement. [file:12]

Une liste vide de prompts produit `[]`, sans charger le modèle. Une liste vide
ou ambiguë de fonctions est rejetée. Un fichier manquant, une syntaxe invalide
ou des types incorrects sont signalés avant l’inférence. Le programme refuse
que la sortie écrase l’un des fichiers d’entrée.

La sortie finale est écrite dans un fichier temporaire puis remplacée
atomiquement. En cas d’erreur, aucune nouvelle sortie partielle n’est présentée
comme un lot réussi ; un ancien fichier de résultats reste éventuellement en
place. Il faut donc vérifier le code de sortie du processus.

## Résultats vérifiés

Les contrôles exécutés sont consignés dans `VALIDATION.txt` : tests de régression,
flake8, mypy avec les options imposées puis en mode strict, syntaxe analysable
comme Python 3.10, contrôles CLI et cohérence du verrou. Les tests ont aussi été
exécutés dans l’environnement des dépendances légères réellement verrouillées,
pas seulement dans l’environnement initial.

Le test facultatif compare l’assemblage des bytes au vrai tokenizer Qwen et suit
un appel complet à travers ses tokens réels. Il vérifie l’encodage et la grammaire,
**pas** la compréhension sémantique du LLM. Les logits pilotés dans les tests
unitaires ne sont jamais utilisés dans l’application.

## Limites restantes

- L’installation complète avec PyTorch et le chargement de Qwen n’ont pas été
  exécutés dans le stockage disponible ; seule une installation partielle et
  la résolution du reste ont été vérifiées.
- Aucun taux réel de sélection/extraction ni temps complet d’inférence n’est
  annoncé : les seuils de 90% et cinq minutes restent à mesurer. [file:12]
- Une grammaire correcte ne garantit pas que le modèle terminera avant le budget
  ni que son choix sémantique sera juste. Un budget épuisé provoque une erreur,
  pas un appel fabriqué pour afficher artificiellement 100% de réussite.
- Le SDK public ne fournit pas de cache KV réutilisable dans les méthodes
  jointes ; l’inférence peut donc recalculer le contexte à chaque token. [file:3]
- Le mapping de vocabulaire vise Qwen byte-level BPE ; le bonus de support
  universel d’autres familles de tokenizers n’est pas revendiqué.
- La validation syntaxique ciblant 3.10 n’équivaut pas à une exécution sur un
  interpréteur Python 3.10 ; les exécutions faites ici utilisent Python 3.12.
- Le login du README et la maîtrise personnelle du code restent à confirmer.

## Vérification locale recommandée

1. Garder une sauvegarde du projet initial ; extraire cette version à part.
2. Remplacer le login du README et examiner le packaging SDK reconstruit.
3. Exécuter `uv sync`, puis `make lint`, `make lint-strict` et `make test`.
4. Lancer la vraie génération et contrôler chaque argument des résultats.
5. Mesurer le lot complet et tester un nouveau catalogue sans noms prédéfinis.
6. Faire une revue avec un pair et expliquer les transitions de grammaire.

Ces dernières vérifications correspondent aux exigences de fiabilité et de
compréhension du sujet ; une archive générée ne les remplace pas. [file:12]
