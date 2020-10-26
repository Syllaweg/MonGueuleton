import fastai
from fastai.vision import *
import torch

from flask import Flask
import requests

import yaml
import json

from io import BytesIO  # Stock data sous forme d'octets, comme les variables
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
    """ charge et décompresse(BytesIO) une image depuis un lien URL avec une requete GET
    Args: 
        url (str): lien url vers une image 
    Return: 
        image : une image 


    utlisation de BytesIO, avec un get pour ouvrir une image:
    https://www.kite.com/python/answers/how-to-read-an-image-data-from-a-url-in-python
    https://pillow.readthedocs.io/en/3.0.x/releasenotes/2.8.0.html
    """

    response = requests.get(url)
    image = open_image(BytesIO(response.content)) 

    return image 