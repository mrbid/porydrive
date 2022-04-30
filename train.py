# James William Fletcher - April 2022
# https://github.com/mrbid/porydrive
import sys
import glob
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from random import seed
from time import time_ns
from sys import exit
from os.path import isdir
from os.path import isfile
from os import mkdir
from tensorflow.keras import backend as K

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# hyperparameters
seed(8008135)
inputsize = 6
outputsize = 2
project = "porydrive_model"
training_iterations = 1
activator = 'tanh'
# layers = 3
layer_units = 32
batches = 32

# training set size
tss = 3341565

# make project directory
if not isdir(project):
    mkdir(project)

# helpers (https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison)
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

##########################################
#   LOAD DATA
##########################################

# load training data
train_x = []
with open("dataset_x.dat", 'rb') as f:
    data = np.fromfile(f, dtype=np.float32)
    train_x = np.reshape(data, [tss, inputsize])

train_y = []
with open("dataset_y.dat", 'rb') as f:
    data = np.fromfile(f, dtype=np.float32)
    train_y = np.reshape(data, [tss, outputsize])

# print(dataset.shape)
# print(dataset)
# exit()

shuffle_in_unison(train_x, train_y)

##########################################
#   TRAIN
##########################################

# construct neural network
model = Sequential()
model.add(Dense(layer_units, activation=activator, input_dim=inputsize))
# for x in range(layers-2):
#     model.add(Dense(layer_units, activation=activator))
model.add(Dense(outputsize, activation='tanh'))

# optim = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer='adam', loss='mean_squared_error')

# train network
st = time_ns()
model.fit(train_x, train_y, epochs=training_iterations, batch_size=batches)
timetaken = (time_ns()-st)/1e+9
print("")
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   EXPORT
##########################################

# save info
if isdir(project):
    # save keras model
    model.save(project + "/keras_model")
    f = open(project + "/model.txt", "w")
    if f:
        f.write(model.to_json())
    f.close()

    # save json model
    f = open(project + "/model.txt", "w")
    if f:
        f.write(model.to_json())
    f.close()

    # save HDF5 weights
    model.save_weights(project + "/weights.h5")

    # save flat weights
    for layer in model.layers:
        if layer.get_weights() != []:
            np.savetxt(project + "/" + layer.name + ".csv", layer.get_weights()[0].flatten(), delimiter=",") # weights
            np.savetxt(project + "/" + layer.name + "_bias.csv", layer.get_weights()[1].flatten(), delimiter=",") # bias

    # save weights for C array
    print("")
    print("Exporting weights...")
    li = 0
    f = open(project + "/" + project + "_layers.h", "w")
    f.write("#ifndef " + project + "_layers\n#define " + project + "_layers\n\n")
    if f:
        for layer in model.layers:
            total_layer_weights = layer.get_weights()[0].flatten().shape[0]
            total_layer_units = layer.units
            layer_weights_per_unit = total_layer_weights / total_layer_units
            #print(layer.get_weights()[0].flatten().shape)
            #print(layer.units)
            print("+ Layer:", li)
            print("Total layer weights:", total_layer_weights)
            print("Total layer units:", total_layer_units)
            print("Weights per unit:", int(layer_weights_per_unit))

            f.write("const float " + project + "_layer" + str(li) + "[] = {")
            isfirst = 0
            wc = 0
            bc = 0
            if layer.get_weights() != []:
                for weight in layer.get_weights()[0].flatten():
                    wc += 1
                    if isfirst == 0:
                        f.write(str(weight))
                        isfirst = 1
                    else:
                        f.write("," + str(weight))
                    if wc == layer_weights_per_unit:
                        f.write(", /* bias */ " + str(layer.get_weights()[1].flatten()[bc]))
                        #print("bias", str(layer.get_weights()[1].flatten()[bc]))
                        wc = 0
                        bc += 1
            f.write("};\n\n")
            li += 1
    f.write("#endif\n")
    f.close()

