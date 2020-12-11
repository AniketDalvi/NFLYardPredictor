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


def clean_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub(' +', ' ', txt)
    txt = txt.strip()
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('retractable', 'rtr.')
    return txt

def transform_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    if 'outdoor' in txt or 'open' in txt:
        return 1
    if 'indoor' in txt or 'closed' in txt:
        return 0
    
    return np.nan

def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans

def str_to_float(txt):
    try:
        return float(txt)
    except:
        return -1

def clean_WindDirection(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = txt.replace('from', '')
    txt = txt.replace(' ', '')
    txt = txt.replace('north', 'n')
    txt = txt.replace('south', 's')
    txt = txt.replace('west', 'w')
    txt = txt.replace('east', 'e')
    return txt

def transform_WindDirection(txt):
    if pd.isna(txt):
        return np.nan
    
    if txt=='n':
        return 0
    if txt=='nne' or txt=='nen':
        return 1/8
    if txt=='ne':
        return 2/8
    if txt=='ene' or txt=='nee':
        return 3/8
    if txt=='e':
        return 4/8
    if txt=='ese' or txt=='see':
        return 5/8
    if txt=='se':
        return 6/8
    if txt=='ses' or txt=='sse':
        return 7/8
    if txt=='s':
        return 8/8
    if txt=='ssw' or txt=='sws':
        return 9/8
    if txt=='sw':
        return 10/8
    if txt=='sww' or txt=='wsw':
        return 11/8
    if txt=='w':
        return 12/8
    if txt=='wnw' or txt=='nww':
        return 13/8
    if txt=='nw':
        return 14/8
    if txt=='nwn' or txt=='nnw':
        return 15/8
    return np.nan

def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0

def new_orientation(angle, play_direction):
    if play_direction == 0:
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            new_angle = 0.0
        return new_angle
    else:
        return angle

sns.set_style('darkgrid')
mpl.rcParams['figure.figsize'] = [15,10]
env = nflrush.make_env()
train = pd.read_csv('train.csv', dtype={'WindSpeed': 'object'})
# transforming stadium type to binary - outdoor or indoor
train['StadiumType'] = train['StadiumType'].apply(clean_StadiumType)
train['StadiumType'] = train['StadiumType'].apply(transform_StadiumType)

#transforming turf to binary - natural or artificial
Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 

train['Turf'] = train['Turf'].map(Turf)
train['Turf'] = train['Turf'] == 'Natural'

#fixing typos in team abbreviations
map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in train['PossessionTeam'].unique():
    map_abbr[abb] = abb

train['PossessionTeam'] = train['PossessionTeam'].map(map_abbr)
train['HomeTeamAbbr'] = train['HomeTeamAbbr'].map(map_abbr)
train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].map(map_abbr)

#Adding fields that indicate whwther home team has posession and whether the field posiotion is with the possession team
train['HomePossesion'] = train['PossessionTeam'] == train['HomeTeamAbbr']
train['Position_Possession'] = train['FieldPosition'] == train['PossessionTeam']
train['HomeField'] = train['FieldPosition'] == train['HomeTeamAbbr']

#one-hot encode offensive formation
off_form = train['OffenseFormation'].unique()
train = pd.concat([train.drop(['OffenseFormation'], axis=1), pd.get_dummies(train['OffenseFormation'], prefix='Formation')], axis=1)
dummy_col = train.columns

#normalizing game clock to time left in quarter
train['GameClock'] = train['GameClock'].apply(strtoseconds)

#Converting height to inches
train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

#Adding BMI
train['PlayerBMI'] = 703*(train['PlayerWeight']/(train['PlayerHeight'])**2)

#Converting handoff and snap time to just time without date
train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

#Calculate difference in snap and handoff time
train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

#Calculate player age
train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
seconds_in_year = 60*60*24*365.25
train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
train = train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)

#Cleaning up windspeed
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
train['WindSpeed'] = train['WindSpeed'].apply(str_to_float)

#Cleaning up wind direction and transform it to numerical feature
train['WindDirection'] = train['WindDirection'].apply(clean_WindDirection)
train['WindDirection'] = train['WindDirection'].apply(transform_WindDirection)

#Assign binary value to play direction
train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x.strip() == 'right')

#Assign binary value to team being home or away
train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')

#Clean up weather and assign it numerical values
train['GameWeather'] = train['GameWeather'].str.lower()
indoor = "indoor"
train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(map_weather)

#Add column that checks whether player involved is rusher
train['IsRusher'] = train['NflId'] == train['NflIdRusher']

#Fix X, orientation, and dir TODO: Understand the math
train['X'] = train.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)
train['Orientation'] = train.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)
train['Dir'] = train.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)

#Add field for yards left to end zone
train['YardsLeft'] = train.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
train['YardsLeft'] = train.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)
train.drop(train.index[(train['YardsLeft']<train['Yards']) | (train['YardsLeft']-100>train['Yards'])], inplace=True)

print("Cleaning done!")

#dropping non affecting features
train = train.sort_values(by=['PlayId', 'Team', 'IsRusher', 'JerseyNumber']).reset_index()
train.drop(['NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1, inplace=True)

#droppping categorical features
cat_features = []
for col in train.columns:
    if train[col].dtype =='object':
        cat_features.append(col)
        
train = train.drop(cat_features, axis=1)

#resstructuring data to coallate all 22 players
train.fillna(-999, inplace=True)
players_col = []
for col in train.columns:
    if train[col][:22].std()!=0:
        players_col.append(col)

X_train = np.array(train[players_col]).reshape(-1, len(players_col)*22)
play_col = train.drop(players_col+['Yards'], axis=1).columns
X_play_col = np.zeros(shape=(X_train.shape[0], len(play_col)))
for i, col in enumerate(play_col):
    X_play_col[:, i] = train[col][::22]
X_train = np.concatenate([X_train, X_play_col], axis=1)
y_train = np.array(train['Yards'])[0::22]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)
batch_size=64

print('Train Pre-processing done!')

#NN
def get_model():
    x = keras.layers.Input(shape=[X_train.shape[1]])
    fc1 = keras.layers.Dense(units=450, input_shape=[X_train.shape[1]])(x)
    act1 = keras.layers.PReLU()(fc1)
    bn1 = keras.layers.BatchNormalization()(act1)
    dp1 = keras.layers.Dropout(0.45)(bn1)
    gn1 = keras.layers.GaussianNoise(0.15)(dp1)
    concat1 = keras.layers.Concatenate()([x, gn1])
    fc2 = keras.layers.Dense(units=600)(concat1)
    act2 = keras.layers.PReLU()(fc2)
    bn2 = keras.layers.BatchNormalization()(act2)
    dp2 = keras.layers.Dropout(0.45)(bn2)
    gn2 = keras.layers.GaussianNoise(0.15)(dp2)
    concat2 = keras.layers.Concatenate()([concat1, gn2])
    fc3 = keras.layers.Dense(units=400)(concat2)
    act3 = keras.layers.PReLU()(fc3)
    bn3 = keras.layers.BatchNormalization()(act3)
    dp3 = keras.layers.Dropout(0.45)(bn3)
    gn3 = keras.layers.GaussianNoise(0.15)(dp3)
    concat3 = keras.layers.concatenate([concat2, gn3])
    predense = keras.layers.Dense(units=200, activation='relu')(concat3)
    dense = keras.layers.Dense(units=64, activation='relu')(predense)
    output = keras.layers.Dense(units=1, activation='linear')(dense)
    model = keras.models.Model(inputs=x, outputs=output)
    return model


def train_model(X_train, y_train, X_val, y_val):
    model = get_model()
    print("model returned")
    model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])
    er = EarlyStopping(patience=20, min_delta=1e-4, monitor='val_loss')
    print("model will now be fitted")
    history = model.fit(X_train, y_train, epochs=200, callbacks=[er], validation_data=[X_val, y_val], batch_size=batch_size)
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(datetime.datetime.now()) + "acc.png")
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(datetime.datetime.now()) + "loss.png")
    model_json = model.to_json()
    filename_model = 'model' + str(datetime.datetime.now()) + '.json'
    with open(filename_model, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    filename_weights = 'model' + str(datetime.datetime.now()) + '.h5'
    model.save_weights(filename_weights)
    filename_pic = 'model' + str(datetime.datetime.now()) + '.png'
    # plot_model(model, to_file=filename_pic)
    return model

rkf = RepeatedKFold(n_splits=5, n_repeats=5)

models = []

for tr_idx, vl_idx in rkf.split(X_train, y_train):
    
    x_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    x_vl, y_vl = X_train[vl_idx], y_train[vl_idx]
    
    model = train_model(x_tr, y_tr, x_vl, y_vl)
    models.append(model)

# i = 0
# for model_new in models:
#     # serialize model to JSON
#     model_json = model_new.to_json()
#     filename_model = 'model' + str(i) + '.json'
#     with open(filename_model, "w") as json_file:
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     filename_weights = 'model' + str(i) + '.h5'
#     model_new.save_weights(filename_weights)
#     filename_pic = 'model' + str(i) + '.png'
#     plot_model(model_new, to_file=filename_pic)
#     i += 1

print("Saved model to disk")

test = pd.read_csv('test.csv', dtype={'WindSpeed': 'object'})
# transforming stadium type to binary - outdoor or indoor
test['StadiumType'] = test['StadiumType'].apply(clean_StadiumType)
test['StadiumType'] = test['StadiumType'].apply(transform_StadiumType)

#transforming turf to binary - natural or artificial
Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 

test['Turf'] = test['Turf'].map(Turf)
test['Turf'] = test['Turf'] == 'Natural'

#fixing typos in team abbreviations
map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in test['PossessionTeam'].unique():
    map_abbr[abb] = abb

test['PossessionTeam'] = test['PossessionTeam'].map(map_abbr)
test['HomeTeamAbbr'] = test['HomeTeamAbbr'].map(map_abbr)
test['VisitorTeamAbbr'] = test['VisitorTeamAbbr'].map(map_abbr)

#Adding fields that indicate whwther home team has posession and whether the field posiotion is with the possession team
test['HomePossesion'] = test['PossessionTeam'] == test['HomeTeamAbbr']
test['Position_Possession'] = test['FieldPosition'] == test['PossessionTeam']
test['HomeField'] = test['FieldPosition'] == test['HomeTeamAbbr']

#one-hot encode offensive formation
test = pd.concat([test.drop(['OffenseFormation'], axis=1), pd.get_dummies(test['OffenseFormation'], prefix='Formation')], axis=1)
# test["OffenseFormation"] = test["OffenseFormation"].apply(lambda x: x if x in off_form else np.nan)

#normalizing game clock to time left in quarter
test['GameClock'] = test['GameClock'].apply(strtoseconds)

#Converting height to inches
test['PlayerHeight'] = test['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

#Adding BMI
test['PlayerBMI'] = 703*(test['PlayerWeight']/(test['PlayerHeight'])**2)

#Converting handoff and snap time to just time without date
test['TimeHandoff'] = test['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
test['TimeSnap'] = test['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

#Calculate difference in snap and handoff time
test['TimeDelta'] = test.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

#Calculate player age
test['PlayerBirthDate'] = test['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
seconds_in_year = 60*60*24*365.25
test['PlayerAge'] = test.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
test = test.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)

#Cleaning up windspeed
test['WindSpeed'] = test['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
test['WindSpeed'] = test['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
test['WindSpeed'] = test['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
test['WindSpeed'] = test['WindSpeed'].apply(str_to_float)

#Cleaning up wind direction and transform it to numerical feature
test['WindDirection'] = test['WindDirection'].apply(clean_WindDirection)
test['WindDirection'] = test['WindDirection'].apply(transform_WindDirection)

#Assign binary value to play direction
test['PlayDirection'] = test['PlayDirection'].apply(lambda x: x.strip() == 'right')

#Assign binary value to team being home or away
test['Team'] = test['Team'].apply(lambda x: x.strip()=='home')

#Clean up weather and assign it numerical values
test['GameWeather'] = test['GameWeather'].str.lower()
indoor = "indoor"
test['GameWeather'] = test['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
test['GameWeather'] = test['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
test['GameWeather'] = test['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
test['GameWeather'] = test['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
test['GameWeather'] = test['GameWeather'].apply(map_weather)

#Add column that checks whether player involved is rusher
test['IsRusher'] = test['NflId'] == test['NflIdRusher']

#Fix X, orientation, and dir TODO: Understand the math
test['X'] = test.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)
test['Orientation'] = test.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)
test['Dir'] = test.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)

#Add field for yards left to end zone
test['YardsLeft'] = test.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
test['YardsLeft'] = test.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)
# test.drop(test.index[(test['YardsLeft']<test['Yards']) | (test['YardsLeft']-100>test['Yards'])], inplace=True)

print("Cleaning done!")

#dropping non affecting features
test = test.sort_values(by=['PlayId', 'Team', 'IsRusher', 'JerseyNumber']).reset_index()
test.drop(['NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1, inplace=True)

#droppping categorical features
cat_features = []
for col in test.columns:
    if test[col].dtype =='object':
        cat_features.append(col)
        
test = test.drop(cat_features, axis=1)

#resstructuring data to coallate all 22 players
test.fillna(-999, inplace=True)
players_col = []
for col in test.columns:
    if test[col][:22].std()!=0:
        players_col.append(col)

X_test = np.array(test[players_col]).reshape(-1, len(players_col)*22)
play_col = test.drop(players_col, axis=1).columns
print(play_col)
X_play_col = np.zeros(shape=(X_test.shape[0], len(play_col)))
for i, col in enumerate(play_col):
    X_play_col[:, i] = test[col][::22]
X_test = np.concatenate([X_test, X_play_col], axis=1)
# y_test = np.zeros(shape=(X_test.shape[0], 199))
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
print(X_test.shape)

print('Test Pre-processing done!')

# models = []
# for i in range(0,25):
#     # load json and create model
#     filename_model = 'model' + str(i) + '.json'
#     json_file = open(filename_model, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     filename_weights = 'model' + str(i) + '.h5'
#     loaded_model.load_weights(filename_weights)
#     models.append(loaded_model)

all_preds = []
j = 0
for model in models:
    pred = model.predict(X_test)
    all_preds.append(pred)
    print("predicting model " + str(j))
    j = j+1

with open('output.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for pred in all_preds:
        employee_writer.writerow(pred)

print("Saved prediction to disk")
# def make_pred(df, sample, env, models):
#     df['StadiumType'] = df['StadiumType'].apply(clean_StadiumType)
#     df['StadiumType'] = df['StadiumType'].apply(transform_StadiumType)
#     df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']
#     df['OffenseFormation'] = df['OffenseFormation'].apply(lambda x: x if x in off_form else np.nan)
#     df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)
#     missing_cols = set( dummy_col ) - set( df.columns )-set('Yards')
#     for c in missing_cols:
#         df[c] = 0
#     df = df[dummy_col]
#     df.drop(['Yards'], axis=1, inplace=True)
#     df['Turf'] = df['Turf'].map(Turf)
#     df['Turf'] = df['Turf'] == 'Natural'
#     df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
#     df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
#     df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
#     df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']
#     # df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']
#     df['HomeField'] = df['FieldPosition'] == df['HomeTeamAbbr']
#     df['GameClock'] = df['GameClock'].apply(strtoseconds)
#     df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
#     df['PlayerBMI'] = 703*(df['PlayerWeight']/(df['PlayerHeight'])**2)
#     df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
#     df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
#     df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
#     df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
#     seconds_in_year = 60*60*24*365.25
#     df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
#     df['WindSpeed'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
#     df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
#     df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
#     df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)
#     df['WindDirection'] = df['WindDirection'].apply(clean_WindDirection)
#     df['WindDirection'] = df['WindDirection'].apply(transform_WindDirection)
#     df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')
#     df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')
#     indoor = "indoor"
#     df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
#     df['GameWeather'] = df['GameWeather'].apply(lambda x: x.lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear').replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
#     df['GameWeather'] = df['GameWeather'].apply(map_weather)
#     df['IsRusher'] = df['NflId'] == df['NflIdRusher']
#     df['X'] = df.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)
#     df['Orientation'] = df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)
#     df['Dir'] = df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)
#     df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
#     df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)
#     df = df.sort_values(by=['PlayId', 'Team', 'IsRusher', 'JerseyNumber']).reset_index()
#     df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1)
#     cat_features = []
#     for col in df.columns:
#         if df[col].dtype =='object':
#             cat_features.append(col)
#     print("Clean")
#     df = df.drop(cat_features, axis=1)
#     df.fillna(-999, inplace=True)
#     X = np.array(df[players_col]).reshape(-1, len(players_col)*22)
#     play_col = df.drop(players_col, axis=1).columns
#     # print(play_col)
#     X_play_col = np.zeros(shape=(X.shape[0], len(play_col)))
#     for i, col in enumerate(play_col):
#         X_play_col[:, i] = df[col][::22]
#     X = np.concatenate([X, X_play_col], axis=1)
#     X = scaler.fit_transform(X)
#     y_pred = np.mean([np.cumsum(model.predict(X), axis=1) for model in models], axis=0)
#     yardsleft = np.array(df['YardsLeft'][::22])
#     for i in range(len(yardsleft)):
#         y_pred[i, :yardsleft[i]-1] = 0
#         y_pred[i, yardsleft[i]+100:] = 1
#     env.predict(pd.DataFrame(data=y_pred.clip(0,1),columns=sample.columns))
#     return y_pred

# for test, sample in tqdm.tqdm(env.iter_test()):
#     models = []
#     for i in range(0,25):
#         # load json and create model
#         filename_model = 'model' + str(i) + '.json'
#         json_file = open(filename_model, 'r')
#         loaded_model_json = json_file.read()
#         json_file.close()
#         loaded_model = model_from_json(loaded_model_json)
#         # load weights into new model
#         filename_weights = 'model' + str(i) + '.h5'
#         loaded_model.load_weights(filename_weights)
#         models.append(loaded_model)
#     make_pred(test, sample, env, models)

