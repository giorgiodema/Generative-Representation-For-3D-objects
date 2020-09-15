
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
from Model.VAE.vae_model import VariationalAutoencoder


BINVOX_LOADER_PATH = os.path.join("C:\\","GL","3DShapeGen","Dataset","executables","Binvox.exe")
BINVOX_LOADER_DIR = os.path.join("C:\\","GL","3DShapeGen","Dataset","executables")
BINVOX_INPUT_TEMP_PATH = os.path.join("C:\\","GL","3DShapeGen","input_tmp.binvox")
BINVOX_OUTPUT_TEMP_PATH = os.path.join("C:\\","GL","3DShapeGen","output_tmp.binvox")
SHAPENET_PATH = os.path.join("G:\\","Documenti","ShapeNetCore.v2")
SUBDIVISIONS = 2
CAT_FILTER = ["table","chair","sofa"]
NUM_CATEGORIES = 3
BATCH_SIZE = 8
BINVOX_DIM = 32
NAME = "learning_pred_and_gen_vector_vae_adam_lr_5_exp5"

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

dataset = dataset.batch(1, drop_remainder=True)
dataset = dataset.map(lambda x:tf.reshape(x,shape=[x.shape[0],x.shape[1],x.shape[2],x.shape[3],1]))
dataset = dataset.map(lambda x:tf.nn.max_pool3d(x, 2*SUBDIVISIONS, 2*SUBDIVISIONS, 'VALID', data_format='NDHWC', name=None))



model  = VariationalAutoencoder(name=NAME,lr=10**-5)
model.initialize(1)
model.load()

# change working dir to launch Binvox.exe
os.chdir(BINVOX_LOADER_DIR)

counter = 0


# loop on dataset
for X in dataset:
    # loop on batch size
    mu,logv = model.encode(X)
    X = tf.reshape(X,shape=[X.shape[1],X.shape[2],X.shape[3]]).numpy().astype(int)
    inp = ''
    while inp!='n':
        print("sampling...")
        z = model.sample(mu,tf.exp(logv))
        Y = model.decode(z)

        Y = tf.reshape(Y[0]>TRESHOLD,shape=[Y.shape[1],Y.shape[2],Y.shape[3]]).numpy().astype(int)

        numpyToBinvox(X,BINVOX_INPUT_TEMP_PATH)
        numpyToBinvox(Y,BINVOX_OUTPUT_TEMP_PATH)

        p = subprocess.Popen([BINVOX_LOADER_PATH,BINVOX_INPUT_TEMP_PATH,BINVOX_OUTPUT_TEMP_PATH])
        inp = input("Hit 'Enter' to show a new sample, 'n' to go to the next element\n")
    print("Next")
    inp=''



