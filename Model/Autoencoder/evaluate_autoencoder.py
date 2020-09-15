
import tensorflow as tf
import subprocess
import time
import os
import sys
import numpy as np
import tensorflow_probability as tfp

import sys
sys.path.append("C:\\GL\\3DShapeGen")
from Dataset.create_binvox_dataset import dataGenerator
from Dataset.create_binvox_dataset import numpyToBinvox
from model import Autoencoder


BINVOX_LOADER_PATH = os.path.join("C:\\","GL","3DShapeGen","Dataset","executables","Binvox.exe")
BINVOX_LOADER_DIR = os.path.join("C:\\","GL","3DShapeGen","Dataset","executables")
BINVOX_INPUT_TEMP_PATH = os.path.join("C:\\","GL","3DShapeGen","input_tmp.binvox")
BINVOX_OUTPUT_TEMP_PATH = os.path.join("C:\\","GL","3DShapeGen","output_tmp.binvox")
SHAPENET_PATH = os.path.join("G:\\","Documenti","ShapeNetCore.v2")
SUBDIVISIONS = 2
CAT_FILTER = ["table","chair","sofa"]
BATCH_SIZE = 8
BINVOX_DIM = 32
NAME = "learning_pred_and_gen_vector_autoencoder_adam_lr_5"

TRESHOLD = 0.4



#
# creates and batch the dataset
# from the generator
#
def generator():
    return dataGenerator(
        SHAPENET_PATH,
        cat_filter=CAT_FILTER,
        generate_binvox=True)

dataset = tf.data.Dataset.from_generator(
    generator,
    output_types=tf.float32,
    output_shapes=[128,128,128])

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.map(lambda x:tf.reshape(x,shape=[x.shape[0],x.shape[1],x.shape[2],x.shape[3],1]))
dataset = dataset.map(lambda x:tf.nn.max_pool3d(x, 2*SUBDIVISIONS, 2*SUBDIVISIONS, 'VALID', data_format='NDHWC', name=None))



model  = Autoencoder(name=NAME,lr=10**-5)
model.initialize(BATCH_SIZE)
model.load()

# change working dir to launch Binvox.exe
os.chdir(BINVOX_LOADER_DIR)

counter = 0


# loop on dataset
for X in dataset:
    # loop on batch size
    Y=model.call(X)
    for i in range(X.shape[0]):
        counter += 1
        print("it:{}".format(counter))
        Y_i = tf.reshape(Y[i]>TRESHOLD,shape=[Y.shape[1],Y.shape[2],Y.shape[3]]).numpy().astype(int)
        X_i = tf.reshape(X[i],shape=[X.shape[1],X.shape[2],X.shape[3]]).numpy().astype(int)

        numpyToBinvox(X_i,BINVOX_INPUT_TEMP_PATH)
        numpyToBinvox(Y_i,BINVOX_OUTPUT_TEMP_PATH)

        p = subprocess.Popen([BINVOX_LOADER_PATH,BINVOX_INPUT_TEMP_PATH,BINVOX_OUTPUT_TEMP_PATH])
        time.sleep(20)
        p.kill()



