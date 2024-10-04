## Quelques infos sur mon repertoire et mon travail
Tout ce qui est en relation avec mon travail se trouve dans le repertoire workspace. Il constitue donc la racine d'acces aux dossiers/fichiers que je vais cités plus tard. 

### Les dossiers 

- Le dossier `notebooks` contient tout les notebooks et codes avec lequel je travaille et tourne la plus part des modeles. Par ailleurs, le dossier `helpers` fournis des fonctions et autres utilitaires utilent a l'entrainement des modeles. 
- Le dossier `datasets` et `csv_files` contiennent respectivement les jeux de donnees et les fichiers csv qui me permettent d'exploiter les datasets. Les donnees que j'ai extraites a partir des videos de veaux a la louve se trouve par contre dans le dossier `/data/data_calves/konrad` 
- Le dossier `models` contient la grande majorites des "best" modeles que j'ai entrainees. On peut aussi en retrouver dans le dossier `training_log` qui contient principalement les logs des entrainements de modeles.
- Le dossier `repos` contient quant a lui, des repertoires git, de quelques modeles avec lesquels je ne pouvais qu'entrainer qu'avec le code source fournis par leur auteur, dont INTR par exemple.

### Dossier `/data/data_calves/konrad` 

Il contient toutes les donnees que j'ai extraites dont deux en particulier:

- `mixed_20s_b0s` qui contient des images et videos de veaux 20s avant le debut de leur visite a la louve, en utilisant YOLO 0
- `mixed_10s_b0s_y7_2024-08-13_17-08-42` qui contient des images et videos de veaux 10s avant le debut de leur visite a la louve, en utilisant YOLO Last
- `spaced_aptm_with_state.csv` qui contient ~6362 visites espacees de plus de 3mn. Il suffit de retirer les classes inutilises pour passer a l'extration.


### Fichier `workspace/notebooks/video_extractions_jobs.ipynb` 

Il sert a faire des extractions de videos/images de veau approchant la louve. Il suffit de modifier dans le notebook les parametres dans la cellule 5 pour l'adapter au besoin. 

