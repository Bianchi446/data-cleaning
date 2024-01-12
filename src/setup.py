import os
import io
import sys
import gzip
import re
import sqlite3
import dbm
from glob import glob
from datetime import datetime, date, timedelta
from pprint import pprint
from math import nan, inf, pi as π, e
import math
from random import seed, choice, randint, sample
from contextlib import contextmanager
from collections import namedtuple
from collections import Counter
from itertools import islice
from textwrap import fill
from dataclasses import dataclass, astuple, asdict, fields
import json
from jsonschema import validate, ValidationError
import simplejson
import requests
import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
import dask
import psycopg2
from sqlalchemy import create_engine

from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV, RFE

from pymongo import MongoClient

from IPython.display import Image as Show
import nltk

# Might use tokenization/stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Do not display warnings
import warnings
warnings.simplefilter('ignore')

# Only show 8 rows from large DataFrame
pd.options.display.max_rows = 8
pd.options.display.min_rows = 8

# A bit of setup for monochrome book; not needed for most work
monochrome = (cycler('color', ['k', '0.5']) * 
              cycler('linestyle', ['-', '-.', ':']))
plt.rcParams['axes.prop_cycle'] = monochrome
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600