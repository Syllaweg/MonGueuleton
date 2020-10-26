import fastai
from fastai.vision import *
import torch

from flask import Flask
import requests

import yaml
import json


from io import BytesIO  # Permet de lire des données Brutes(nos images)
# plus généralement de gérer différent type de données I/O -> https://docs.python.org/3/library/io.html
from typing import List, Dict, Union, ByteString, Any # Syntaxe pour Typer ses varb
import sys  # module system


"""
# need pip install pyyaml
with open("src/config.yaml", 'r') as stream:
    try:
        APP_CONFIG = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
"""


app = Flask(__name__) # instance de Flask, (si name == main -> app run)


def charge_model(path='.', model_name="model.pkl"):
    """ Charge le model entrainé, model.pkl

    """

    model = load_learner(path, fname=model_name)
    return model


def charge_img_url(url):
    """ charge et décompresse(BytesIO) une image brute depuis un lien URL avec une requete GET
    Args: 
        url (str): lien url vers une image 
    Return: 
        image : une image 

    - utlisation de BytesIO, avec un get pour ouvrir une image:
    https://www.kite.com/python/answers/how-to-read-an-image-data-from-a-url-in-python
    https://pillow.readthedocs.io/en/3.0.x/releasenotes/2.8.0.html
    """

    response = requests.get(url)
    image = open_image(BytesIO(response.content)) 

    return image 


def charge_img_brute(raw):
    """ Charge et ouvre des images depuis des données brute en bits
    Args: 
        byte : image brute en bits
    Return:
        image : une image utilisable
    """

    image = open_image(BytesIO(raw))
    
    return image 


def prediction(image, n = 3):
    """ Analyse une image avec le modèle entrainé, et en ressort une prédiction de classe
    Args:
        image : image à traiter
        n (int): nombre de prédiction à afficher, ici les 3 premières
    Return:
        class_predicton (str): Classe prédite pour l'img
        predictions (lst): liste des probabilités des prédictions de classes pour une img
    """

    class_prediction, predict_idx, outputs = model.predict(image) # model.predict, renvoie un Tuple de 3 élem: 
                                                                    # 1 la pred passé par l'activation et la fct de perte
                                                                    # 2 la pred décodé
                                                                    # 3 la pred décodé en ulisant les transform appliqué au DataLoaders
    predict_proba = outputs / sum(outputs)
    predict_proba = predict_proba.tolist()

    predictions = []

    for img_class, retour, proba in zip(model.data.classes, outputs.tolist(), predict_proba):
        output = round(output, 1 )
        proba = round(proba, 2)
        predictions.append({"Classe": img_class.replace("_"," "), "Sortie": retour, "probalité": proba})

    predictions = sorted(predictions, key=lambda x: x["output"], reverse=True)
    predictions = predictions[0 : n] # affichera les n premiers resultats (3 ici)

    return {"Classe": str(class_prediction), "Predictions": predictions}

