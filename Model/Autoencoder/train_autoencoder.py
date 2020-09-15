import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# Local Moduels
import sys
sys.path.append("C:\\GL\\3DShapeGen")
from Dataset.create_binvox_dataset import dataGenerator,numpyToBinvox
from model import Autoencoder





#
# parameters
#
SHAPENET_PATH = os.path.join("G:","Documenti","ShapeNetCore.v2")
CACHE_PATH = os.path.join("G:","Documenti","Cache","cache")
NAME = "learning_pred_and_gen_vector_autoencoder_adam_lr_5"
RESULTS = "C:\\GL\\3DShapeGen\\Results\\learning_pred_and_gen_vector"
STATE = os.path.join(RESULTS,NAME+"state.dat")
CAT_FILTER=["table","chair","sofa"]
SUBDIVISIONS = 2
BINVOX_DIM = 32
BATCH_SIZE = 8
NUM_CATEGORIES = 3
EPOCHS = 200

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
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.cache(filename=CACHE_PATH)

""" INSPECT THE DATASET
import subprocess
import time
BINVOX_LOADER_PATH = os.path.join("C:\\","GL","3DShapeGen","Dataset","executables","Binvox.exe")
BINVOX_LOADER_DIR = os.path.join("C:\\","GL","3DShapeGen","Dataset","executables")
BINVOX_INPUT_TEMP_PATH = os.path.join("C:\\","GL","3DShapeGen","input_tmp.binvox")
BINVOX_OUTPUT_TEMP_PATH = os.path.join("C:\\","GL","3DShapeGen","output_tmp.binvox")
os.chdir(BINVOX_LOADER_DIR)

for x in dataset.take(5):
    for i in range(BATCH_SIZE):

        res = tf.reshape(x[i],shape=(32,32,32)).numpy().astype(int)
        numpyToBinvox(res,BINVOX_INPUT_TEMP_PATH)
        p = subprocess.Popen([BINVOX_LOADER_PATH,BINVOX_INPUT_TEMP_PATH])
"""
#
# initializes the model and the
# statistics for the training
#
model  = Autoencoder(name=NAME,lr=10**-5)
model.initialize(BATCH_SIZE)

start_epoch = 0
min_loss = sys.maxsize
it = 0
losses_per_it = []
losses_per_epoch = []
loss = 0

#
# resume the training, if the program is
# started with the argument RESUME, the state file
#
if len(sys.argv)==2 and sys.argv[1]=="RESUME":
    if not os.path.exists(STATE):
        raise Exception("Impossible to resume training. The previous state is not available")
    model.load()
    with open(STATE,"r") as f:
        lines = f.readlines()
        start_epoch = int(lines[0])
        min_loss = float(lines[1])
        for x in lines[2].split(","):
            try:
                losses_per_epoch.append(float(x))
            except Exception:
                continue


#
# training loop
#
try:
    for e in range(start_epoch,EPOCHS):
        for X in dataset:
            it +=1
            loss = model.learn(X).numpy()[0]

            losses_per_it.append(loss.astype(np.float16))
            #plt.suptitle(NAME)
            #plt.xlabel("it")
            #plt.ylabel("loss")
            #plt.plot(losses_per_it,'b')
            #plt.savefig(os.path.join(RESULTS,model.name+"_ep"+str(e)+"_losses_per_epoch"))
            #plt.clf()

            print("Epoch:{}/{} it:{} Loss:{}".format(e,EPOCHS,it,loss))

        loss = round(sum(losses_per_it)/len(losses_per_it),4)
        losses_per_it = []
        it = 0
        if loss < min_loss:
            print("loss decreased {} --> {}, saving model".format(min_loss,loss))
            min_loss = loss
            model.save()

        losses_per_epoch.append(loss.astype(np.float16))
        plt.suptitle(NAME)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(losses_per_epoch,'b')
        plt.savefig(os.path.join(RESULTS,model.name+"_losses"))
        plt.clf()

#
# Saving the state to resume training
# State file has one line for each value:
#       -> current_epoch
#       -> min_loss
#       -> losses per epoch
#
except KeyboardInterrupt:
    print("Saving State")
    with open(STATE,"w") as f:
        print(str(e),file=f)
        print(str(min_loss),file=f)
        for x in losses_per_epoch:
            print(str(x),file=f,end=",")
        print("",file=f)



