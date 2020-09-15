import tensorflow as tf
import numpy as np
import itertools
import pickle
import os

RES_PATH = "C:\\GL\\3DShapeGen\\Results\\learning_pred_and_gen_vector"

class Autoencoder:

    def __init__(self,name="Autoencoder",
            lr=10**-6,
            loss=tf.losses.binary_crossentropy,
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)):
        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss=loss
        self.initialized = False
        self.name=name
        self.conv1 = tf.keras.layers.Conv3D(48,4,kernel_initializer=initializer,activation="relu",name="conv1")    # 29 --
        self.conv2 = tf.keras.layers.Conv3D(96,4,kernel_initializer=initializer,activation="relu",name="conv2")    # 26
        self.conv3 = tf.keras.layers.Conv3D(176,4,kernel_initializer=initializer,activation="relu",name="conv3")   # 23 --
        self.conv4 = tf.keras.layers.Conv3D(256,4,kernel_initializer=initializer,activation="relu",name="conv4")   # 20
        self.conv5 = tf.keras.layers.Conv3D(320,4,kernel_initializer=initializer,activation="relu",name="conv5")   # 17 --
        self.conv6 = tf.keras.layers.Conv3D(384,4,kernel_initializer=initializer,activation="relu",name="conv6")   # 14
        self.conv7 = tf.keras.layers.Conv3D(320,4,kernel_initializer=initializer,activation="relu",name="conv7")   # 11 --
        self.conv8 = tf.keras.layers.Conv3D(256,4,kernel_initializer=initializer,activation="relu",name="conv8")   # 8
        self.fc = tf.keras.layers.Dense(512,kernel_initializer=initializer,activation=None,name="fc")

        self.convt1 = tf.keras.layers.Conv3DTranspose(256,4,kernel_initializer=initializer,activation="relu",name="convt1")    # 11
        self.convt2 = tf.keras.layers.Conv3DTranspose(384,4,kernel_initializer=initializer,activation="relu",name="convt2")    # 14
        self.convt3 = tf.keras.layers.Conv3DTranspose(320,4,kernel_initializer=initializer,activation="relu",name="convt3")    # 17
        self.convt4 = tf.keras.layers.Conv3DTranspose(256,4,kernel_initializer=initializer,activation="relu",name="convt4")    # 20
        self.convt5 = tf.keras.layers.Conv3DTranspose(176,4,kernel_initializer=initializer,activation="relu",name="convt5")    # 23
        self.convt6 = tf.keras.layers.Conv3DTranspose(96,4,kernel_initializer=initializer,activation="relu",name="convt6")     # 26
        self.convt7 = tf.keras.layers.Conv3DTranspose(48,4,kernel_initializer=initializer,activation="relu",name="convt7")     # 29
        self.convt8 = tf.keras.layers.Conv3DTranspose(1,4,kernel_initializer=initializer,activation="sigmoid",name="convt8")   # 32
        self.layers = [ self.conv1,
                        self.conv2,
                        self.conv3,
                        self.conv4,
                        self.conv5,
                        self.conv6,
                        self.conv7,
                        self.conv8,
                        self.fc,
                        self.convt1,
                        self.convt2,
                        self.convt3,
                        self.convt4,
                        self.convt5,
                        self.convt6,
                        self.convt7,
                        self.convt8]

    def initialize(self,bs):
        self.initialized=True
        t = tf.convert_to_tensor(np.ndarray((bs,32,32,32,1)),dtype=tf.float32)
        self.input = tf.Variable(tf.convert_to_tensor(np.ndarray((bs,32,32,32,1)),dtype=tf.float32))
        o = self.encode(t)
        o = self.decode(o)
        self.variables = []
        for l in self.layers:
            for x in l.trainable_variables:
                self.variables.append(x)
    

    def encode(self,X):
        self.input.assign(X)
        o = self.conv1(self.input)
        o = self.conv2(o)
        o = self.conv3(o)
        o = self.conv4(o)
        o = self.conv5(o)
        o = self.conv6(o)
        o = self.conv7(o)
        o = self.conv8(o)                                                               #[bs, 8, 8, 8, 256]
        o = tf.reshape(o,[o.shape[0],o.shape[1]*o.shape[2]*o.shape[3]*o.shape[4]])
        o = self.fc(o)
        return o

    def decode(self,X):
        o = tf.reshape(X,[X.shape[0],8,8,8,1])                                          #[bs, 8, 8, 8, 1]
        o = self.convt1(o)
        o = self.convt2(o)
        o = self.convt3(o)
        o = self.convt4(o)
        o = self.convt5(o)
        o = self.convt6(o)
        o = self.convt7(o)
        o = self.convt8(o)
        o = tf.reshape(o,[o.shape[0],o.shape[1],o.shape[2],o.shape[3],1])
        return o


    def call(self,X):
        return self.decode(self.encode(X))

    @tf.function()
    def learn(self,X):
        with tf.GradientTape() as tape:
            tape.watch(self.input)
            o = self.call(X)
            L = self.loss(tf.reshape(self.input,[self.input.shape[0],32**3]),tf.reshape(o,[o.shape[0],32**3]))
        g = tape.gradient(L,self.variables)
        self.optimizer.apply_gradients(zip(g,self.variables))
        return L

    # return all the weights of the trainable layers
    def get_weights(self):
        weights = {}
        for l in self.layers:
            weights[l.name] = l.get_weights()
        return weights

    # sets all the weights of the trainable layers
    def set_weights(self,w):
        for l in self.layers:
            l.set_weights(w[l.name])

    def save(self):
        assert(self.initialized)
        d = self.get_weights()
        with open(os.path.join(RES_PATH,self.name+"_weights"),"wb") as f:
            pickle.dump(d,f)

    def load(self):
        assert(self.initialized)
        with open(os.path.join(RES_PATH,self.name+"_weights"),"rb") as f:
            d = pickle.load(f)
            self.set_weights(d)
        