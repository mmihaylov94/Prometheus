from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime
from time import time


#Main neural network function
def network(combination, learning_rate, epochs, batches, seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    start = time()
    
    checkpoint_path = "mnist-{}-{}-{}-{}-{}.ckpt".format(combination,learning_rate,epochs,batches,seed)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=0, period=1)
    
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./logs/run-{}".format(now), write_graph=True, update_freq="batch")
    
    X_train, y_train, X_test, y_test = get_mnist_data()
    
    if combination == 1 or combination == 3:
        X_train, X_test = reshape_mnist_for_CNN(X_train, X_test)
    
    model = build_model(combination, learning_rate)
    
    model.fit(X_train, y_train, batch_size=batches,
              epochs=epochs, verbose=1, validation_split=0.1,
              shuffle="batch", callbacks = [cp_callback, tensorboard])
    
    end = time()
    

    score = model.evaluate(X_test, y_test, verbose=0)
    
    print("Test accuracy: {}%.".format(score[1]*100))
    print("The model took {} minutes and {} seconds to train.".format(np.int32((end-start)//60), np.int32((end-start)%60)))

#Function to reshape data to fit to a CNN
def reshape_mnist_for_CNN(X_train, X_test):
    X_train_reshaped = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    return X_train_reshaped, X_test_reshaped

#Load and return the data with a split of 60,000 training instances and 10,000 testing instances
def get_mnist_data():
    ((X_train, y_train), (X_test, y_test)) = tf.keras.datasets.mnist.load_data()

    X_train = X_train/np.float32(255)
    y_train = y_train.astype(np.int32)

    X_test = X_test/np.float32(255)
    y_test = y_test.astype(np.int32)
    
    return X_train, y_train, X_test, y_test

#Build and return the correct model depending on the combination
def build_model(combination, learning_rate):
    model = tf.keras.Sequential()
    
    if combination == 1 or combination == 3:
        model = network_one(combination, model, activation=tf.nn.relu)
    
    if combination == 2 or combination == 4:
        model = network_two(combination, model, activation=tf.nn.relu)
    
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)    
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=adam,
                  metrics=['accuracy'])      
    return model
    
#CNN network architecture
def network_one(combination, model, activation):
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), 
                     activation=activation, 
                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2))) 
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), 
                     activation=activation)) 
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=activation))
    if combination == 3:
        model.add(tf.keras.layers.Dense(128, activation=activation))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
    return model      

#MLP network architecture
def network_two(combination, model, activation):
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(512, activation=activation))
    model.add(tf.keras.layers.Dense(256, activation=activation))
    if combination == 4:
        model.add(tf.keras.layers.Dense(128, activation=activation))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    return model

#Method for restoring models
def restore_model(checkpoint_path):
    combination = np.int(checkpoint_path.split("-")[1])
    learning_rate = np.float32(checkpoint_path.split("-")[2])
    X_train, y_train, X_test, y_test = get_mnist_data()
    
    if combination == 1 or combination == 3:
        X_train, X_test = reshape_mnist_for_CNN(X_train, X_test)
    
    model = build_model(combination, learning_rate)
    model.load_weights(checkpoint_path)  
    score = model.evaluate(X_test, y_test)
    print("Model restored! \nTest accuracy: {}%.".format(score[1]*100))

#Checks if the input parameter is a float
def check_param_is_float(param, value):

    try:
        value = float(value)
    except:
        print("{} must be float".format(param))
        quit(1)
    return value    

#Checks if the input parameter is an integer
def check_param_is_int(param, value):
    
    try:
        value = int(value)
    except: 
        print("{} must be integer".format(param))
        quit(1)
    return value

# 4 Combinations
# Combination 1: CNN with 1 hidden layer
# Combination 2: MLP with 2 hidden layers
# Combination 3: CNN with 2 hidden layers
# Combination 4: MLP with 3 hidden layers

# Run a neural network with the given parameters
#network(1, 0.005, 10, 64, 12345)

#Restore the specified model and check the test accuracy
#checkpoint_path = "mnist-4-0.001-20-32-12345.ckpt"
#restore_model(checkpoint_path)    

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")

    args = arg_parser.parse_args()

    combination = check_param_is_int("combination", args.combination)
    learning_rate = check_param_is_float("learning_rate", args.learning_rate)
    epochs = check_param_is_int("epochs", args.iterations)
    batches = check_param_is_int("batches", args.batches)
    seed = check_param_is_int("seed", args.seed)

    network(combination, learning_rate, epochs, batches, seed)