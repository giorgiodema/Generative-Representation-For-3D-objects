import os
import numpy as np
import sys
import json
import tensorflow as tf
import pickle
from PIL import Image
from multiprocessing.dummy import Pool
import time


#
# convert a binvox file
# into a numpy array
#
def binvoxToNumpy(path):
    binary_data = None
    dim = None
    with open(path,"rb") as f:
        for b in f:
            s = b.decode('utf-8')
            s = s.split(" ")
            if s[0].strip() == "dim":
                dim = int(s[1])
            if s[0].strip() == "data":
                binary_data = f.read()
                break
    array = np.zeros(shape=(dim*dim*dim),dtype=int)
    idx = 0
    for v,l in zip(binary_data[0::2],binary_data[1::2]):
        for i in range(idx,idx+l):
            array[i] = v
        idx = idx + l
    array = np.reshape(array,(dim,dim,dim))
    return array

#
# convert a numpy array into a
# binvox file
#
def numpyToBinvox(array,path):
    dim = array.shape[0]
    array = np.reshape(array,(dim*dim*dim))
    b = bytearray()
    b.extend(createHeader(dim))
    idx = 0
    while idx < array.shape[0]:
        v = array[idx]
        count = 0
        while idx < array.shape[0] and array[idx] == v and count < 255:
            count += 1
            idx += 1
        b.extend([v,count])
        
    with open(path,"wb") as f:
        f.write(b)


def createHeader(dim):
    s = "#binvox 1\ndim {} {} {}\ndata\n".format(str(dim),str(dim),str(dim))
    b = bytearray()
    b.extend(map(ord, s))
    return b


def shapeNetInfo(shapenet_path,num_categories="every"):
    with open(os.path.join(shapenet_path,"taxonomy.json"),"r") as f:
        s = f.read()
        taxonomy = json.loads(s)
        categories = [(taxonomy[i]["numInstances"],taxonomy[i]["name"],taxonomy[i]['synsetId']) for i in range(len(taxonomy))]
        categories.sort(key=lambda x:x[0],reverse=True)
        if num_categories=="every":
            return categories
        elif type(num_categories)==int:
            return categories[0:num_categories]
        else:
            raise Exception("Invalid Argument: num_categories can be either 'every' or an integer")



def __parallelProduceSnapshots(args):
    path = args[0]
    imp = args[1]

    imp = os.path.join(path,imp)
    im = Image.open(imp)
    im = im.crop((0,56,1440,1080))
    im = im.crop((208,0,1232,1024))
    im = im.resize((128,128))
    return np.asarray(im,dtype=np.float32)

def dataGenerator(  shapenet_path,
                    num_categories="every",
                    cat_filter=[],
                    generate_path=False,
                    generate_binvox=False,
                    generate_encoding=False,
                    generate_label=False,
                    generate_snapshots=False,
                    split_snapshots=False,
                    encoding_name="encoding.pk"):
    #
    # select the first "num_categories" 
    # categories with more samples
    #
    categories = shapeNetInfo(  
        shapenet_path,
        num_categories=num_categories)
    #
    # categories have multiple names, 
    # select the first one for each
    # category
    #
    categories = list(  
        map(
            lambda a: (a[0],a[1].split(",")[0],a[2]),
            categories))
    #
    # remove the categories that do not match the filter
    #
    if len(cat_filter)>0:
        categories = list(filter(lambda a: a[1] in cat_filter,categories))
        if len(cat_filter)!=len(categories):
            raise Exception("Invalid Argument: a category in cat_filter does not exist")
    for c in categories:
        print(c)

    dirs = list(map(lambda x:x[2],categories))
    #
    # compute the average length of each category to balance the dataset,
    # at most avg_len element for each category will be considered
    #
    avg_len = sum(list(map(lambda x: int(x[0]),categories)))/len(dirs)

    #
    # consider the ith element for each
    # category
    #
    for i in range(int(avg_len)):
        for j,d in enumerate(dirs):
            d = os.path.join(shapenet_path,d)
            cat = categories[j][1]
            # skip taxonomy file
            if os.path.isfile(d):
                continue
            subs = os.listdir(d)
            # if i can't take the ith element of directory d
            # go to the next directory
            if i >= len(subs):
                continue
            sample_dir = subs[i]
            sample_dir = os.path.join(d,sample_dir)
            # check if the sample dir path exists
            if not os.path.exists(sample_dir):
                continue
            #
            # BINVOX
            #
            if generate_binvox:
                # convert the model into a numpy array and return
                model_path = os.path.join(sample_dir,"models","model_normalized.solid.binvox")
                if not os.path.exists(model_path):
                    continue
                array = binvoxToNumpy(model_path)
            #
            # SNAPSHOTS
            #
            if generate_snapshots:
                path = os.path.join(sample_dir,"snapshots")
                if not os.path.exists(path):
                    continue
                # create the snapshots tensor
                args = [(path,imp) for imp in os.listdir(path)]
                p = Pool(32)
                o = p.map(__parallelProduceSnapshots,args)
                o = np.stack(o,axis=0)
            #
            # ENCODING
            #
            if generate_encoding:
                encoding_path = os.path.join(sample_dir,encoding_name)
                if not os.path.exists(encoding_path):
                    continue
                # create the encoding tensor
                ser = tf.io.read_file(encoding_path)
                encoding = tf.io.parse_tensor(ser,tf.float32)
                #with open(encoding_path,"rb") as f:
                #    encoding = pickle.load(f)

            #
            # yield elements
            #
            result = []
            if generate_path:
                result.append(sample_dir)

            if generate_binvox:
                result.append(array)

            if generate_encoding:
                result.append(encoding)

            if generate_label:
                result.append(cat)

            if not generate_snapshots:
                t = tuple(result)
                if len(t)>1:
                    yield t
                else:
                    yield t[0]
            else:
                if not split_snapshots:
                    result.append(o/255.0)
                    t = tuple(result)
                    if len(t)>1:
                        yield t
                    else:
                        yield t[0]
                else:
                    for k in range(72):
                        t = tuple(result + [o[k]/255])
                        if len(t)>1:
                            yield t
                        else:
                            yield t[0]


"""
SHAPENET_PATH = os.path.join("G:","Documenti","ShapeNetCore.v2")
CAT_FILTER=["table","chair","sofa"]
ENCODING_NAME = "learning_pred_and_gen_vector_vae_adam_lr_5_exp5_embedding.dat"

g = dataGenerator(  SHAPENET_PATH,
                    num_categories="every",
                    cat_filter=CAT_FILTER,
                    generate_path=False,
                    generate_binvox=False,
                    generate_encoding=True,
                    generate_label=False,
                    generate_snapshots=True,
                    split_snapshots=False,
                    encoding_name=ENCODING_NAME)

for (enc,snapshots) in g:
    pass

"""