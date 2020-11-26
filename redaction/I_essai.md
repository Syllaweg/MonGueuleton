# Premier jet, projet CdO  
***
<br>

## ***Résumé***  
La Vision par Ordinateur est devenu, depuis 60 ans, un domaine de recherche très prisé. Les traitements optimisent les interfaces des différents produits commerciaux développés par l’industrie, en les rendant plus attractifs, plus utiles, plus intéressants et surtout omniprésents.  
Les applications centrées sur le consommateur exigent de plus en plus que l’analyse de ces images soit robuste à toute la gamme de bruit réel et à d’autres conditions de distorsion. Cependant, reconnaître de manière fiable des objets ou des actions dans des environnements réalistes reste toujours un défi.  

Notre tâche consistera à concevoir un système “Intelligent” permettant d’adapter un réseau de neurones à la reconnaissance d’image de plats et de nourriture, ce système sera ensuite déployé via une pipeline dans une application qui permettra à des utilisateurs d’interagir avec lui.  
Dans la première étape nous procédons par itération afin de tester et évaluer différentes architectures et hyper-paramètres lors de l'entraînement de notre modèle, pour sélectionner le plus cohérent pour notre tâche. Pour ce faire nous utiliserons des outils comme Pytorch, Fastai, Numpy et des architectures Resnet.



<br>

## ***Table des matières***

- Résumé (abstract)
- Table de matières
- Avant propos: parcours Simplon/Orange

#### **Introduction**
- 1.1 L'intelligence Artificiel aujourd'hui
  - 1.1.1 L’interaction Homme-Machine et ses données
- 1.2 Contexte du projet 
- 1.4 Problématique
- 1.5 Objectifs et domaine d'application

#### **Vision par ordinateur et classification d'images**
- 2.1 Introduction au projet 
    - 2.1.1 Le jeux de données
    - 2.1.2 les technologies: *Pytorch, FastAI, Resnet*
- 2.2 Analyse des données
- 2.3 Architecture du modèle *Resnet*
- 2.4 Évaluation de différents modèles
  - 2.4.1 Augmentation des données
  - 2.4.2 Pré-traitement
  - 2.4.3 Évaluation 
- 2.5 Amélioration du modèle
    - 2.5.1 Nettoyage de la base de données
    - 2.5.2 Ajout de la classe *"Inconnue"*
- 2.6 résultats



<br>

## ***Introduction***

### 1.1 l'Intelligence Artificiel aujourd'hui  

L’intelligence artificielle vit aujourd’hui un renouveau incroyable, portée depuis la fin des années 80 par des géants comme *IBM* et son système *Deep Blue* [1] qui a rendu visible pour le grand public les progrès de l’apprentissage automatique par les machines. Ce qui nous a menée jusqu’à aujourd’hui à un regain d'intérêt autour des différentes techniques d’IA qui arrivent petit à petit dans notre quotidien, amenée par de grands acteurs comme *Apple* avec son *Face ID* [2], mais aussi *IBM, Google*, …, également par de petites structures dynamiques et spécialisées comme des startup.  

La démocratisation croissante des moteurs neuronaux depuis les années 2010 avec des *Graphical Processing Unit* toujours plus puissants, des bibliothèques toujours plus nombreuses qui permettent d’exploiter des données également disponible en toujours plus grand nombre, augmente de manière de manière exponentiel les métiers et les activités potentiellement touchés (Sécurité, Education, Finance, Domotique, Systèmes d’Information...).  
Dans d’autres domaines encore, la vision par ordinateur qui est certainement aujourd'hui le domaine d’application des réseaux de neurones le plus aboutie, est utilisée pour apprendre à des machines à identifier leur environnement afin de s'y déplacer. Il faut aussi compter avec des applications sociétales spectaculaires comme les voitures autonomes et l’imagerie médicale. Avec par exemple la détection de cellules cancéreuses, il sera certainement possible d’automatiser la détection des zones infectés et de prédire les traitements qui auront le plus de chances d’aboutir. 

#### 1.1.1 L’interaction Homme-Machine et ses données

Les moyens de collectes d’information c’étant depuis ces 20 dernières années considérablement améliorés.
Les données collectées par les acteurs privés ou étatiques via les téléphone mobiles qui nous suivent désormais partout, internet ou le nombre de services proposé est toujours plus grand, dans des procédés industriels capable aujourd’hui de connecté et de récupérer des données depuis toujours plus de machines. 

Force est de constater que désormais les techniques traditionnelles d’analyse ne suffisent plus pour traiter ce flot immense de données, qui reste donc en partie inexploitées.  
Et c’est dans ce cadre que l’IA telle que nous l’entendons aujourd’hui vient répondre pour partie à ces problématiques de traitement des données, grâce à des algorithmes que les chercheurs tentent de développer, d’améliorer afin d’augmenter notre potentiel d’exploitation de ces données.

Pour aller plus loin on peut imaginer qu’avec le développement de l’*Internet des Objets*, les interactions *machine-machine* et *Homme-machine* soit toujours plus nombreuses et la nécessité de traiter correctement ces données de plus en plus diverses soit donc toujours plus grandes.  
On peut prendre en exemple les récentes avancées dans la reconnaissance vocale (aujourd'hui 150 mots/minute contre 20-50 mots/minute au clavier) qui ouvre de  nombreuses nouvelles voies, avec un environnement de machine connectés entre elles qui nécessite pour un usage harmonieux, un traitement des données, une *Intelligence Artificiel*, capable de répondre à ces nouveaux usages spécifiques.


