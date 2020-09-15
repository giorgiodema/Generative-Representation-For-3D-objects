import os
import sys
import pickle
import tensorflow as tf
from itertools import islice
from tqdm import tqdm
import numpy as np
from multiprocessing.dummy import Pool
sys.path.append("C:\\GL\\3DShapeGen")
from Dataset.create_binvox_dataset import dataGenerator
from Dataset.create_binvox_dataset import binvoxToNumpy
#
# Change this import to change the model
# you want to use to produce embeddings
#
from model import Autoencoder as Model

def __writeDiskMapFn(args):
    path = args[0]
    emb  = args[1]
    name = args[2]

    path = os.path.join(path,name+"_embedding.dat")
    tf.io.write_file(
        path,
        tf.io.serialize_tensor(emb))
#
# Produce the embeddings for all the elements matching the filter, the encoding
# of the ith element is saved in the root directory of the ith element with the
# name: NAME_embedding.dat
#
def saveEncodings(
            SHAPENET_PATH = os.path.join("G:\\","Documenti","ShapeNetCore.v2"),
            cat_filter=["table","chair","sofa"],
            SUBDIVISIONS = 2,
            NAME = "learning_pred_and_gen_vector_autoencoder_adam_lr_5",
            BATCH_SIZE = 8,
            TMP = "C:\\GL\\3DShapeGen\\tmp\\pathslist.pk"):
    #
    # Instantiate model
    #
    model  = Model(name=NAME,lr=10**-5)
    model.initialize(BATCH_SIZE)
    model.load()
    #
    # Load in advance the paths of all the elements
    # we need to get the embeddings
    #
    if os.path.exists(TMP):
        with open(TMP,"rb") as f:
            paths = pickle.load(f)
    else:
        paths = [x for x in dataGenerator(  SHAPENET_PATH,
                                            cat_filter=cat_filter,
                                            generate_path=True)]
        with open(TMP,"wb") as f:
            pickle.dump(paths,f)

    #
    # batch the elements and for each batch produce a tensor of binvoxes
    #
    batches = []
    for i in range(0,len(paths),BATCH_SIZE):
        batches.append(paths[i:i+BATCH_SIZE])
    for batch in tqdm(batches):
        models_paths = [ os.path.join(x,"models","model_normalized.solid.binvox") for x in batch]
        binvoxs = []
        for x in models_paths:
            if os.path.exists(x):
                binvoxs.append(binvoxToNumpy(x))
            else:
                binvoxs.append(np.zeros(shape=(128,128,128)))
        binvoxs = tf.convert_to_tensor(binvoxs,dtype=tf.float32)
        binvoxs =tf.reshape(binvoxs,shape=[binvoxs.shape[0],binvoxs.shape[1],binvoxs.shape[2],binvoxs.shape[3],1])
        binvoxs = tf.nn.max_pool3d(binvoxs, 2*SUBDIVISIONS, 2*SUBDIVISIONS, 'VALID', data_format='NDHWC', name=None)
        #
        # Produce the embeddings, the embeddings has shape: (bs,embedding_dim)
        #
        embeddings = model.encode(binvoxs)
        #
        # parallel write of the elements to disk
        #
        args = [(path,emb,NAME) for path,emb in zip(batch,embeddings)]
        p = Pool(BATCH_SIZE)
        p.map(__writeDiskMapFn,args)



