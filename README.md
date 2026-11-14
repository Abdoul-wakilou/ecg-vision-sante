# ECG Vision Santé : lire des ECG avec l'IA, du signal à la photo

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Licence MIT](https://img.shields.io/badge/Licence-MIT-green)
![Statut](https://img.shields.io/badge/Statut-en%20cours-orange)
![Learning in public](https://img.shields.io/badge/Learning-in%20public-blueviolet)

**FR :** Projet d'apprentissage, mené en public, sur l'analyse automatique d'électrocardiogrammes (ECG) par traitement du signal et vision par ordinateur. On part du signal numérique brut et on va jusqu'à la photo d'un ECG papier abîmé. Les étapes se suivent : détecter les battements, classer les pathologies, mesurer la robustesse, reconstruire le signal à partir d'une image, puis vérifier l'équité du modèle entre sous-groupes de patients.

**EN:** A learning-in-public project on automated electrocardiogram (ECG) analysis with signal processing and computer vision, from raw digital signals to photos of damaged paper ECGs. The steps follow one another: beat detection, disease classification, robustness testing, image-to-signal digitization, and fairness checks across patient subgroups.

> **Projet d'apprentissage, pas un outil clinique.** Voir l'[avertissement médical](#12-avertissement-médical).

## Sommaire

1. [Contexte : pourquoi ce projet](#1-contexte--pourquoi-ce-projet)
2. [Objectif et questions de recherche](#2-objectif-et-questions-de-recherche)
3. [Comment le projet avance (learning in public)](#3-comment-le-projet-avance-learning-in-public)
4. [Un ECG en une minute](#4-un-ecg-en-une-minute)
5. [Feuille de route](#5-feuille-de-route)
6. [Démarche : les étapes d'un projet de machine learning](#6-démarche--les-étapes-dun-projet-de-machine-learning)
7. [Données](#7-données)
8. [Installation](#8-installation)
9. [Structure du projet](#9-structure-du-projet)
10. [Résultats](#10-résultats)
11. [Limites connues dès le départ](#11-limites-connues-dès-le-départ)
12. [Avertissement médical](#12-avertissement-médical)
13. [Auteur](#13-auteur)
14. [Références](#14-références)
15. [Licence](#15-licence)

## 1. Contexte : pourquoi ce projet

Au Bénin, comme dans beaucoup de pays d'Afrique de l'Ouest, l'ECG est un examen courant, mais les cardiologues restent rares. Beaucoup d'ECG n'existent **que sur papier** : ils sont imprimés, pliés, rangés dans un dossier, parfois photographiés au téléphone. Les modèles d'IA publiés sont presque toujours entraînés sur des signaux numériques propres, issus d'hôpitaux européens ou américains.

Ce projet fait suite à mon mémoire de Master, [CardioPredict](https://github.com/Abdoul-wakilou/cardio_predict_model), qui prédisait le risque cardiovasculaire à partir de **données tabulaires**. Une revue postérieure à la soutenance y a révélé une **fuite de données** qui surestimait les performances (AUC de 0,980 annoncée, 0,844 une fois corrigée), ainsi qu'une chute nette des performances d'un centre hospitalier à l'autre. Les deux leçons que j'en garde guident ce nouveau projet :

1. **séparer les données correctement**, par patient et par centre, avant toute chose ;
2. **se demander si le modèle tient hors de ses données d'entraînement.**

Ici, je passe des tableaux aux **signaux** et aux **images**.

## 2. Objectif et questions de recherche

**Objectif :** apprendre, étape par étape et de façon reproductible, les méthodes d'analyse d'ECG par IA, en documentant chaque résultat, y compris les échecs.

Questions auxquelles le projet veut répondre, chiffres à l'appui :

| # | Question |
|---|---|
| Q1 | De combien un découpage aléatoire des battements surestime-t-il les performances, par rapport à un découpage par patient ? |
| Q2 | Un CNN qui « regarde » l'image d'un ECG fait-il aussi bien qu'un modèle qui lit directement le signal ? |
| Q3 | Quelle dégradation de l'image (rotation, pli, photo, écriture manuscrite) fait le plus chuter le modèle, et l'augmentation de données suffit-elle à compenser ? |
| Q4 | Peut-on reconstruire un signal exploitable à partir d'une image d'ECG avec du traitement d'image classique ? |
| Q5 | Le modèle est-il aussi performant et aussi bien calibré chez les femmes et les hommes, les jeunes et les âgés ? |

## 3. Comment le projet avance (learning in public)

Je ne suis pas cardiologue : j'apprends à lire un ECG en même temps que je construis les modèles. Le dépôt est public **dès le premier jour** et montre le travail tel qu'il avance.

- **Un notebook par étape**, numérotés dans l'ordre (`01_`, `02_`...). Chacun commence par les questions auxquelles il répond et se termine par mes conclusions écrites.
- **Explorer avant de modéliser.** Aucun modèle n'est entraîné avant d'avoir compris les données, défini la métrique et fixé le découpage entraînement / test.
- **Le jeu de test n'est jamais regardé pendant le réglage.** Les hyperparamètres se choisissent sur un jeu de validation séparé.
- **Les erreurs restent visibles.** Une erreur découverte en cours de route est corrigée et expliquée, pas effacée.
- **Reproductible.** Les graines aléatoires sont fixées, les versions des bibliothèques sont figées, et un script télécharge les données.

La [feuille de route](#5-feuille-de-route) indique où j'en suis.

## 4. Un ECG en une minute

Un ECG enregistre l'**activité électrique du cœur** grâce à des électrodes posées sur la peau. Un ECG standard a **12 dérivations** : ce sont 12 « angles de vue » sur le même cœur.

Chaque battement dessine trois ondes :

```text
        R
        |
   P    |      T
  /\   |\    /\
_/  \_/  \__/  \___
         Q S
```

| Onde | Ce qui se passe dans le cœur |
|---|---|
| **P** | les oreillettes se contractent |
| **QRS** | les ventricules se contractent (le grand pic) |
| **T** | les ventricules se rechargent |

Sur le papier, la grille sert de règle. À la vitesse standard de 25 mm/s :
- horizontalement, 1 petit carreau = 0,04 s et 1 grand carreau = 0,2 s ;
- verticalement, 10 mm = 1 mV.

Pour estimer la fréquence cardiaque, divise 300 par le nombre de grands carreaux entre deux pics R.

**Les données de ce projet ne sont pas des images.** PTB-XL et MIT-BIH contiennent des **signaux**, c'est-à-dire des suites de nombres en millivolts. Un ECG de PTB-XL à 100 Hz est un tableau de 1000 × 12 valeurs : 10 secondes sur 12 dérivations. Les images, c'est ce projet qui les fabrique (exercices 2 à 4). Seul le dataset Mendeley contient de vraies photos.

## 5. Feuille de route

| # | Étape | Compétence | Données | Statut |
|---|---|---|---|---|
| 1 | Détecter les battements et classer les arythmies | Traitement du signal, validation inter-patients | MIT-BIH | 🟡 en cours : exploration |
| 2 | Transformer les signaux en images et classer avec un CNN | Vision par ordinateur, transfer learning | PTB-XL | ⚪ à venir |
| 3 | Tester la robustesse aux images abîmées | Évaluation hors distribution | PTB-XL + ECG-Image-Kit | ⚪ à venir |
| 4 | Numériser une image d'ECG en signal | Traitement d'image (grille, segmentation, tracé) | PTB-XL + ECG-Image-Kit | ⚪ à venir |
| 5 | Expliquer le modèle et vérifier son équité | IA responsable, Grad-CAM, calibration | PTB-XL (âge, sexe) | ⚪ à venir |
| Bonus | Démo mobile : photo d'ECG → analyse | Flutter + API Flask | Modèles des étapes 2 et 4 | ⚪ à venir |

<details>
<summary>Détail de l'étape 1 (MIT-BIH)</summary>

- [ ] Explorer les données : enregistrements, annotations, classes, patients (`01_exploration_mitbih.ipynb`)
- [ ] Filtrer le signal (passe-bande Butterworth 0,5 à 40 Hz)
- [ ] Détecter les pics R (Pan-Tompkins codé à la main, comparé à `neurokit2`), avec sensibilité et VPP à ± 150 ms
- [ ] Regrouper les annotations en 5 classes AAMI (N, S, V, F, Q) et exclure les enregistrements avec pacemaker
- [ ] Extraire chaque battement (fenêtre autour du pic R + intervalles RR)
- [ ] Random Forest puis CNN 1D, avec un découpage aléatoire puis un découpage par patient (DS1/DS2, de Chazal et al., 2004)

**Livrable :** tableau sensibilité / VPP / F1 par classe pour les deux découpages.

</details>

## 6. Démarche : les étapes d'un projet de machine learning

Chaque étape de la feuille de route suit le même ordre. Il est volontairement lent au début, car la plupart des erreurs graves, comme la fuite de données de mon mémoire, se commettent avant même d'entraîner un modèle.

| # | Étape | Question à se poser | Où |
|---|---|---|---|
| 1 | **Comprendre le sujet** | Qu'est-ce qu'un ECG ? Qu'est-ce qu'on cherche à détecter, et pourquoi c'est utile ? | sections 1 et 4 de ce README |
| 2 | **Obtenir les données** | D'où viennent-elles ? Sous quelle licence ? Comment les télécharger de façon reproductible ? | `scripts/download_data.py` |
| 3 | **Explorer** | Quelle forme ont les données ? Combien d'exemples par classe ? Comment se répartissent-ils entre patients ? Y a-t-il des pièges ? | `notebooks/01_exploration_*.ipynb` |
| 4 | **Définir le résultat attendu** | Quelle métrique a du sens ? Quel découpage entraînement / test ? Quel score de référence (baseline) faut-il battre ? | conclusions de l'exploration |
| 5 | **Prétraiter** | Filtrage, normalisation, sélection des dérivations, regroupement des classes : chaque choix est justifié par l'exploration | notebooks suivants |
| 6 | **Modèle simple d'abord** | Un Random Forest bien évalué vaut mieux qu'un réseau profond mal évalué | |
| 7 | **Modèle plus riche** | CNN 1D, puis CNN sur images | |
| 8 | **Évaluer une seule fois sur le test** | Les chiffres finaux viennent du jeu de test, jamais utilisé avant | |
| 9 | **Analyser les erreurs** | Où le modèle se trompe-t-il, et chez qui ? | |

**Où j'en suis :** étapes 1 à 3 pour MIT-BIH.

## 7. Données

Tous les datasets sont publics et gratuits. **Les données ne sont pas dans ce dépôt** : le script `scripts/download_data.py` les télécharge depuis PhysioNet.

| Dataset | Contenu | Format | Taille | Licence | Utilisé à l'étape |
|---|---|---|---|---|---|
| [MIT-BIH Arrhythmia v1.0.0](https://physionet.org/content/mitdb/1.0.0/) | 48 enregistrements de 30 min, 47 sujets, environ 110 000 battements annotés par des cardiologues | WFDB, 360 Hz, 2 dérivations | 77 Mo | ODC Attribution 1.0 | 1 |
| [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | 21 799 ECG de 10 s sur 12 dérivations, 18 869 patients ; 5 superclasses diagnostiques (NORM, MI, STTC, CD, HYP), multi-étiquettes ; âge, sexe ; 10 folds fournis | WFDB, 100 Hz et 500 Hz | 1,8 Go (ZIP) | CC BY 4.0 | 2 à 5 |
| [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit) | Outil, pas un dataset : génère des images d'ECG réalistes (papier, plis, rotation, écriture) à partir des signaux | Python | – | BSD-3-Clause | 3, 4 |
| [PhysioNet Challenge 2024](https://moody-challenge.physionet.org/2024/) | Énoncé officiel « numérisation et classification d'images d'ECG », avec un script d'évaluation | Code | – | – | 4 |
| [ECG Images dataset of Cardiac Patients](https://data.mendeley.com/datasets/gwbz3fsgp8/2) | Vraies images d'ECG 12 dérivations (Pakistan) | Images | – | CC BY 4.0 | Bonus |

### Les 5 superclasses de PTB-XL

| Classe | Signification |
|---|---|
| NORM | ECG normal |
| MI | infarctus du myocarde |
| STTC | anomalies du segment ST et de l'onde T |
| CD | troubles de la conduction (blocs) |
| HYP | hypertrophie (paroi du cœur épaissie) |

## 8. Installation

### Prérequis

- Python 3.10 ou plus récent (développé avec Python 3.12)
- Environ 2 Go d'espace disque pour PTB-XL à 100 Hz, ou 5 Go avec la version à 500 Hz
- Facultatif : un GPU, ou Google Colab (gratuit) pour les étapes 2 à 5

### En local

```bash
git clone https://github.com/Abdoul-wakilou/ecg-vision-sante.git
cd ecg-vision-sante

python -m venv .venv
# Windows (PowerShell) :
.\.venv\Scripts\Activate.ps1
# Linux / macOS :
source .venv/bin/activate

pip install -r requirements.txt
```

### Télécharger les données

```bash
python scripts/download_data.py mitdb    # étape 1 : MIT-BIH, 77 Mo
python scripts/download_data.py ptbxl    # étapes 2 à 5 : PTB-XL à 100 Hz (ZIP de 1,8 Go)
```

Les données arrivent dans `data/`, un dossier exclu de Git. Si le téléchargement est interrompu, relance la même commande : il reprend là où il s'était arrêté. Pour PTB-XL, l'option `--full` garde aussi les signaux à 500 Hz.

### Lancer les notebooks

```bash
jupyter notebook notebooks/
```

ou ouvre directement les fichiers `.ipynb` dans VS Code. Dans ce cas, choisis le noyau `.venv` en haut à droite du notebook (*Select Kernel*). Sinon, VS Code peut utiliser un autre Python où `wfdb` n'est pas installé.

## 9. Structure du projet

```text
notebooks/       un notebook par étape, numérotés dans l'ordre
  01_exploration_mitbih.ipynb
scripts/
  download_data.py   téléchargement reproductible des datasets
src/             code réutilisable, extrait des notebooks quand il se stabilise (à venir)
results/         figures et tableaux finaux (à venir)
data/            données téléchargées, non versionnées
requirements.txt versions figées des bibliothèques
```

## 10. Résultats

*Les résultats seront ajoutés ici au fur et à mesure, avec le chiffre clé et la figure principale de chaque étape.*

| Étape | Résultat clé | Figure |
|---|---|---|
| 1 | à venir | |
| 2 | à venir | |
| 3 | à venir | |
| 4 | à venir | |
| 5 | à venir | |

## 11. Limites connues dès le départ

- **Aucune donnée africaine.** MIT-BIH vient de Boston (années 1970-1980), PTB-XL d'Allemagne (1989-1996). Rien ne garantit que les résultats se transposent à des patients béninois.
- **Images synthétiques.** Les images des étapes 3 et 4 sont générées par ordinateur. Elles imitent un ECG papier, mais l'écart avec de vraies photos sera mesuré au bonus, sur le dataset Mendeley.
- **Petit nombre de patients dans MIT-BIH.** Avec 47 sujets, les classes rares viennent souvent d'un ou deux patients : les performances par classe seront incertaines.
- **Pas de validation clinique.** Aucun cardiologue n'a évalué les sorties du projet à ce stade.

## 12. Avertissement médical

Ce dépôt est un **projet personnel d'apprentissage**. **Il ne pose pas de diagnostic et ne remplace pas l'avis d'un professionnel de santé.** Il n'a fait l'objet d'aucune certification comme dispositif médical ni d'aucune validation clinique. N'utilisez pas ses résultats pour prendre une décision concernant un patient.

## 13. Auteur

**Abdoul-Wakilou TIGA**, ingénieur en génie logiciel, en préparation d'un projet de thèse en IA pour la santé.

- E-mail : [abdoulwakiloutiga@gmail.com](mailto:abdoulwakiloutiga@gmail.com)
- GitHub : [github.com/Abdoul-wakilou](https://github.com/Abdoul-wakilou)
- LinkedIn : [linkedin.com/in/abdoul-wakilou-tiga](https://linkedin.com/in/abdoul-wakilou-tiga)
- Kaggle : [kaggle.com/abdoulwakiloutiga](https://kaggle.com/abdoulwakiloutiga)
- Site : [abdevfreelance.com](https://abdevfreelance.com)

Projet précédent : **CardioPredict**, prédiction du risque cardiovasculaire, mémoire de Master en Génie Logiciel (IFRI, Université d'Abomey-Calavi) : [modèle et API](https://github.com/Abdoul-wakilou/cardio_predict_model), [application mobile](https://github.com/Abdoul-wakilou/cardio_predict_mobile_app).

## 14. Références

Les licences des datasets exigent de les citer.

**Datasets**

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology 20(3):45-50, 2001.
- Wagner P, Strodthoff N, Bousseljot RD, Kreiseler D, Lunze FI, Samek W, Schaeffter T. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7:154, 2020.
- Wagner P, Strodthoff N, Bousseljot RD, Samek W, Schaeffter T. *PTB-XL, a large publicly available electrocardiography dataset* (version 1.0.3). PhysioNet, 2022. https://doi.org/10.13026/kfzx-aw45
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, et al. *PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals.* Circulation 101(23):e215-e220, 2000.
- Shivashankara KK, et al. *ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization.* Physiological Measurement, 2024.

**Méthodes**

- Pan J, Tompkins WJ. *A real-time QRS detection algorithm.* IEEE Transactions on Biomedical Engineering 32(3):230-236, 1985.
- de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE Transactions on Biomedical Engineering 51(7):1196-1206, 2004.
- Strodthoff N, Wagner P, Schaeffter T, Samek W. *Deep learning for ECG analysis: benchmarks and insights from PTB-XL.* IEEE Journal of Biomedical and Health Informatics 25(5):1519-1528, 2021.

## 15. Licence

Le **code** de ce dépôt est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE).

Les **données** ne sont pas redistribuées ici. Elles restent soumises à leurs licences respectives (ODC-By 1.0 pour MIT-BIH, CC BY 4.0 pour PTB-XL), indiquées à la [section 7](#7-données).
