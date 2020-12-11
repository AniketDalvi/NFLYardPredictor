import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
from kaggle.competitions import nflrush
import tqdm
import re
from string import punctuation
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf
from keras.models import model_from_json
from sklearn.model_selection import RepeatedKFold
import csv

prediction = []
with open('output.csv', mode='r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    prediction=[row for idx, row in enumerate(readCSV) if idx == 25]

prediction = np.array(prediction)
prediction = prediction.flatten()
prediction = prediction.astype(float)
print(prediction.shape)

actual = []
with open('nfl_big_data_bowl_2020_stage1_solution.csv', mode='r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        actual.append(row[1])

actual = np.array(actual).astype(float)
print(actual.shape)

diff = actual - prediction
diff_2 = diff * diff
rmse = (np.sum(diff_2)/ 3439)**(0.5)
print(rmse)
abs = np.sum(np.absolute(diff))/3439
print(abs)

# load json and create model
filename_model = 'model.json'
json_file = open(filename_model, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
filename_weights = 'model.h5'
loaded_model.load_weights(filename_weights)

plot_model(loaded_model, to_file='model.png')