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