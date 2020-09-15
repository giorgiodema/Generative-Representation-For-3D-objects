import os
import numpy as np
import tensorflow as tf
from functools import reduce
import subprocess

import sys
sys.path.append("C:\\GL\\3DShapeGen")
from Dataset.create_binvox_dataset import dataGenerator,numpyToBinvox
from Model.my_model.vae_model import VariationalAutoencoder as Model

INTERPOLATE="SAME" # "DIFFERENT | SAME"
SHAPENET_PATH = os.path.join("G:","Documenti","ShapeNetCore.v2")
CAT_FILTER = ["table","chair","sofa"]
MODEL_NAME = "learning_pred_and_gen_vector_vae_adam_lr_5_exp5"
ENCODING_NAME = "learning_pred_and_gen_vector_vae_adam_lr_5_exp5_embedding.dat"
INTERPOLATION_STEPS = 10
TRESHOLD = 0.4
TEMP_PATH = "C:\\GL\\3DShapeGen\\tmp"
BINVOX_LOADER_PATH = os.path.join("C:\\","GL","3DShapeGen","Dataset","executables","Binvox.exe")
BINVOX_LOADER_DIR = os.path.join("C:\\","GL","3DShapeGen","Dataset","executables")
SUBDIVISIONS = 2

#
# instantiate model
#
model = Model(name=MODEL_NAME)
model.initialize(1)
model.load()

#
# instantiate generator
#
g = dataGenerator(
    SHAPENET_PATH,
    cat_filter=CAT_FILTER,
    generate_binvox=True,
    generate_encoding=True,
    generate_label=True,
    encoding_name=ENCODING_NAME)

# change working dir to launch Binvox.exe
os.chdir(BINVOX_LOADER_DIR)

while True:
    elems = []
    if INTERPOLATE =="DIFFERENT":
        cats = []
        cond = False
        #
        # loop until in elems there is exactly one
        # element for each different category in
        # cat_filter
        #
        while not cond:
            try:
                e = next(g)
                if e[2] not in cats:
                    cats.append(e[2])
                    elems.append(e)
            except StopIteration:
                print("Finish")
                exit(0)
            
            cond = list(map(lambda x:x in cats,CAT_FILTER))
            cond = reduce(lambda x,y: x and y,cond)
    elif INTERPOLATE=="SAME":
        try:
            e1 = next(g)
            e2 = next(g)
            while e1[2]!=e2[2]:
                e2 = next(g)
            elems = [e1,e2]
        except StopIteration:
            print("Finish")
            exit(0)



    steps = np.arange(0.0, 1.0, 1.0/INTERPOLATION_STEPS, dtype=np.float32)
    #
    # create all possible pair of different elements for
    # interpolation
    #
    couples = [(x,y) for x in elems for y in elems if not (x[0]==y[0]).all() ]

    for c in couples:
        #
        # i {1,2}   -> starting and arrival of interpolation
        # bi        -> binvox
        # ei        -> encoding
        # li        -> label
        #
        b1,e1,l1 = c[0]
        b2,e2,l2 = c[1]
        # each enconding contains n_samples samples,
        # take only the first sample
        e1 = e1[0]
        e2 = e2[0]

        b1 = tf.reshape(
            tf.nn.max_pool3d(
                tf.reshape(
                    tf.convert_to_tensor(b1,dtype=tf.float32),
                    (1,128,128,128,1)), 
                2*SUBDIVISIONS, 2*SUBDIVISIONS, 'VALID', data_format='NDHWC', name=None),
            (32,32,32)).numpy().astype(int)

        b2 = tf.reshape(
            tf.nn.max_pool3d(
                tf.reshape(
                    tf.convert_to_tensor(b2,dtype=tf.float32),
                    (1,128,128,128,1)), 
                2*SUBDIVISIONS, 2*SUBDIVISIONS, 'VALID', data_format='NDHWC', name=None),
            (32,32,32)).numpy().astype(int)
            
        print("{}   --->    {}".format(l1,l2))

        #
        # view start and arrival
        #
        binvox_start_path = os.path.join(TEMP_PATH,"start.binvox")
        binvox_end_path = os.path.join(TEMP_PATH,"end.binvox")
        numpyToBinvox(b1,binvox_start_path)
        numpyToBinvox(b2,binvox_end_path)
        exe_start = subprocess.Popen([BINVOX_LOADER_PATH,binvox_start_path])
        exe_end = subprocess.Popen([BINVOX_LOADER_PATH,binvox_end_path])
        input()

        #
        # view all intermediate steps
        #
        for i,t in enumerate(steps):
            print("it:{}".format(i))
            e = (1 - t) * e1 + t * e2
            b = tf.reshape(
                    model.decode(
                        tf.reshape(tf.convert_to_tensor(e),(1,512)))>TRESHOLD,
                    (32,32,32)).numpy().astype(int)
            binvox_path = os.path.join(TEMP_PATH,"{}.binvox".format(i))
            numpyToBinvox(b,binvox_path)
            exe = subprocess.Popen([BINVOX_LOADER_PATH,binvox_path])
            a=input("Hit Enter to continue, s to skip\n")
            if a =="s":
                break
            #
            # kill processes and remove temporary data
            #
            exe.kill()
            os.remove(binvox_path)

        #
        # kill processes and remove temporary data
        #        
        os.remove(binvox_start_path)
        os.remove(binvox_end_path)
        exe_start.kill()
        exe_end.kill()


