import re
from math import ceil
from collections import defaultdict
import pandas as pd
import graphlab
import sqlite3
from tabulate import tabulate
from pymongo import MongoClient
# import sklearn
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import random
import gc
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


sf = graphlab.SFrame.read_csv('all_ratings.csv')

plt.hist(sf['player_rating'])
