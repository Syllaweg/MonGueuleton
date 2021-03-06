from fastai import *
import fastai
from fastai.vision import *
import torch

import flask
from flask import Flask
import requests

import yaml
import json

from io import BytesIO  # Permet de lire des données Brutes(nos images)
# plus généralement de gérer différent type de données I/O -> https://docs.python.org/3/library/io.html
import sys  # module system


with open("src/config.yaml", 'r') as stream:
    APP_CONFIG = yaml.full_load(stream)


app = Flask(__name__) # instance de Flask, (si name == main -> app run)


# ---------------------------------------------------------------------------- #
#                                   Fonctions                                  #
# ---------------------------------------------------------------------------- #

def charge_model(path=".", model_name="model.pkl"):
    """ Charge le model entrainé, model.pkl
    Args:
        path (str): chemin vers le fichier du model
        model_name (str): nom du fichier contenant le modele
    Return:
        model: retourne le modèle entrainé prêt à être utilisé
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
    rep = requests.get(url)
    image = open_image(BytesIO(rep.content)) 

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


def model_run(image, n=3):
    """ Analyse une image avec le modèle entrainé, et en ressort une prédiction de classe
    Args:
        image : image à traiter
        n (int): nombre de prédiction à afficher, ici les 3 premières
    Return:
        class_predicton (str): Classe prédite pour l'img
        predictions (lst): liste des probabilités des prédictions de classes pour une img
    """
    class_prediction, pred_idx, outputs = model.predict(image) # model.predict, renvoie un Tuple de 3 élem: 
                                                                    # 1 la pred passé par l'activation et la fct de perte
                                                                    # 2 la pred décodé
                                                                    # 3 la pred décodé en ulisant les transform appliqué au DataLoaders
    predict_proba = outputs / sum(outputs)
    predict_proba = predict_proba.tolist()

    predictions = []

    for img_class, sortie, proba in zip(model.data.classes, outputs.tolist(), predict_proba):
        output = round(sortie, 1)
        proba = round(proba, 2)
        predictions.append({"classe": img_class.replace("_", " "), "sortie": sortie, "proba": proba})

    predictions = sorted(predictions, key=lambda x: x["sortie"], reverse=True)
    predictions = predictions[0 : n] # affichera les n premiers resultats (3 ici)

    return {"classe": str(class_prediction), "predictions": predictions}


# ---------------------------------------------------------------------------- #
#                                     Flask                                    #
# ---------------------------------------------------------------------------- #

@app.route('/api/classifieur', methods=['POST', 'GET'])
def charge_fichier():
    """ Recupere une image depuis un URL ou un fichier et la passe dans le modele 
    Return:
        ret (json): la prediction dans un fichier JSON
    """
    # va chercher une image via une URL avec une requete GET
    if flask.request.method == 'GET':
        url = flask.request.args.get("url")
        image = charge_img_url(url) # utlise notre fct pour requeter et charger l'image
    else: 
    # Si l'img est fourni directement comme un fichier (pas via un URL)
        bytes = flask.request.files["file"].read()
        image = charge_img_brute(bytes)
    
    ret = model_run(image) # Fait la prediction sur l'img

    return flask.jsonify(ret)


@app.route('/api/classes', methods=['GET'])
def classes():
    """ Récupère avec GET le nom des classes prédites
    Return:
        classes prédites str dans un fichier JSON
    """
    class_predite = sorted(model.data.classes)
    
    return flask.jsonify(class_predite)


@app.route('/Bon', methods=['GET'])
def test():
    """
    Test la connection à l'api
    """
    return "jour!"


@app.route('/config')
def config():
    return flask.jsonify(APP_CONFIG)


# Gestion des mises en Cache
#if app.config["DEBUG"]:
@app.after_request
def http_header(reponse):
    """ Modification des en-tetes de reponse HTTP, pour éviter les reponse stocké en Cache

    https://developer.mozilla.org/fr/docs/Web/HTTP/Headers
    https://www.codeflow.site/fr/article/spring-security-cache-control-headers
    https://perso.liris.cnrs.fr/pierre-antoine.champin/2019/progweb-python/cours/cm4.html
    https://medium.com/@maskaravivek/how-to-add-http-cache-control-headers-in-flask-34659ba1efc0
    """
    reponse.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0" # spécifie des directives pour les mécanismes de mise en cache dans les requêtes et les réponses.
    reponse.headers["Expires"] = 0 # La date et l'heure après lesquelles la réponse est considérée périmé
    reponse.headers["Pragma"]= "no-cache" # rétrocompatibilité avec les caches HTTP/1.0 où l'en-tête Cache-Control n'est pas présent.
    reponse.cache_control.max_age = 0 # La durée en secondes passée par l'objet dans un cache proxy.

    return reponse


@app.route('/<path:path>')
def static_file(path):
    if '.js' in path or '.css' in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file("index.html")


@app.route('/')
def root():
    return app.send_static_file('index.html')


def before_requets():
    app.jinja_env.cache = {}



model = charge_model("models") # args -> path vers le modele


if __name__ == "__main__":
    port = os.environ.get('PORT', 5000)

    if 'prepare' not in sys.argv:
        app.jinja_env.auto_reload = True
        app.config["TEMPLATES_AUTO_RELOAD"] = True
        app.run(debug=False, host="0.0.0.0", port=port)


