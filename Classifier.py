import numpy
import pandas
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.optimizers import adam
from keras.models import model_from_json

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("/home/kartheek/PycharmProjects/Thesis/venv/data/lineitemSF1.tbl", header=None, delimiter='|')
#We need to create 2 extra columns in the dataframe: positon and sortKey
#To assign position you will have to iterate through the dataframe, for each OrderKey you find:
#If you havent seen it before you will give it a random position, and save the pair OrderKey-PositionGiven in a hashmap
#If you have seen it before, then you will give the position that you have in the hash map.

#To assign sortKey you will also go through the dataFrame and do the following:
#You start from 0...
#Each time the key changes, you increase your counter by a random number between 1 and 15.
#If the key does not change you give it the value of the counter.

dataset = dataframe.values
print(dataset.shape)

X = dataset[:,0].astype(float)
POSITIONS = []
HASH_TABLE={}
RandomPosUsed= set()
for item in X:
    if item in HASH_TABLE:
        POSITIONS.append(HASH_TABLE[item])
    else:
        randomnumber= numpy.random.randint(0,15000000000)
        while(randomnumber in RandomPosUsed):
            randomnumber= numpy.random.randint(0,15000000000)
        RandomPosUsed.add(randomnumber)
        HASH_TABLE[item]=randomnumber
        POSITIONS.append(randomnumber)

RegressorY=[]
Y=[]
currentKey=X[0]
counter=0
realcounter=0
partition="0"
for item in X:
    realcounter+=1
    if not item==currentKey:
        counter+=numpy.random.randint(0,15)
        currentKey=item
        if partition=="0" and realcounter>len(X)/4:
            partition="1"
        elif partition=="1" and realcounter>len(X)/2:
            partition="2"
        elif partition=="2" and realcounter>3*len(X)/4:
            partition="3"
    RegressorY.append(counter)
    Y.append(partition)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(X[0])
print(dummy_y[0])
#print(X.shape)
#print(dummy_y)

optimizer= adam(lr=0.001)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=1, init='uniform', activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.save_weights('model.h5')
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=128, verbose=2)


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#to save the weights into hdf5py format
#model.save_weights('model.h5')

'''
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
'''

