#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import socket
import pickle
import tensorflow as tf
from tensorflow import keras


SAVE_PATH = 'model'


def get_network():
    num_filters = [24,32,64,128] 
    pool_size = (2, 2) 
    kernel_size = (3, 3)  
    input_shape = (60, 41, 2)
    num_classes = 2
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(24, kernel_size,
                padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(32, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    
    model.add(keras.layers.Conv2D(64, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    
    model.add(keras.layers.Conv2D(128, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  

    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(1e-4), 
        loss=keras.losses.SparseCategoricalCrossentropy(), 
        metrics=["accuracy"])
    return model


# In[9]:


HOST = '192.168.56.1'
PORT = 65433

HEADERSIZE = 10

X, y = [], []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    
    end_trans = False
    
    s.bind((HOST, PORT))
    s.listen(5)
   
    full_msg = b''
    new_msg = True

    socket, address = s.accept()
    while True: 
        msg = socket.recv(51200)
        if msg == b'':
            socket.close()
            break
        if new_msg:
            msglen = int(msg[:HEADERSIZE])
            new_msg = False
        full_msg += msg
        if msglen == len(full_msg) - HEADERSIZE:
            data = pickle.loads(full_msg[HEADERSIZE:])
            new_msg = True
            if not data:
                break
            socket.sendall(b'recieved message')
            full_msg = b''
            features = pickle.loads(data["features"])
            features = np.concatenate(np.array(features), axis=0) 
            
            labels = np.concatenate(pickle.loads(data["labels"]), axis=0)
            X.append(features)
            y.append(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

X_train= np.concatenate(X_train, axis = 0).astype(np.float32)
y_train = np.concatenate(y_train, axis = 0).astype(np.float32)

model = get_network()
model.fit(X_train, y_train, epochs = 30, batch_size = 24, verbose = 1)

model.save(SAVE_PATH)
y_true, y_pred = [], []
for x, y in zip(X_test, y_test):
    print(x.shape)
    avg_p = np.argmax(np.mean(model.predict(x), axis = 0))
    y_pred.append(avg_p) 
    y_true.append(np.unique(y)[0]) 
print(accuracy_score(y_true, y_pred))    

# In[ ]:




