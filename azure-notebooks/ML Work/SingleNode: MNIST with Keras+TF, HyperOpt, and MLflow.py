# Databricks notebook source
# MAGIC %md # MNIST with Keras, HyperOpt, and MLflow
# MAGIC 
# MAGIC **Purpose**: Trains a simple ConvNet on the MNIST dataset using Keras + HyperOpt using [Databricks Runtime for Machine Learning](https://databricks.com/blog/2018/06/05/distributed-deep-learning-made-simple.html)

# COMMAND ----------

# MAGIC %md ### Load Keras & Tensorflow libraries

# COMMAND ----------

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import math
import time
import tempfile

# Use TensorFlow Backend
import tensorflow as tf
tf.set_random_seed(42) # For reproducibility

# COMMAND ----------

# MAGIC %md ### Load MLFlow and HyperOpt libraries

# COMMAND ----------

# Including MLflow
import mlflow
import mlflow.keras
import os
print("MLflow Version: %s" % mlflow.__version__)

# Configure Databricks MLflow environment
EXPERIMENT_NAME = "mab_mnist_multi_hpopt"
mlflow.set_experiment(EXPERIMENT_NAME)

from datetime import datetime
RUN_NAME = "mnist_kerastf_GPU_" + datetime.now().strftime('%d%h%y')
print RUN_NAME

# Include HyperOpt
from hyperopt import fmin, hp, tpe, STATUS_OK



# COMMAND ----------

# MAGIC %md ### Load, split, and pre-process train/test datasets

# COMMAND ----------

# Image Datasets

# input image characteristics
img_rows, img_cols = 28, 28
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# COMMAND ----------

# MAGIC %md ### Define CNN network

# COMMAND ----------

def create_model(hpo):
  
  model = Sequential()
  
  # Convolution Layer
  model.add(Conv2D(32, kernel_size=(int(hpo['kernel']), int(hpo['kernel'])),
                 activation='relu',
                 input_shape=input_shape)) 
  
  # Convolution layer
  model.add(Conv2D(64, (int(hpo['conv']), int(hpo['conv'])), activation='relu'))
  
  # Pooling with stride
  model.add(MaxPooling2D(pool_size=(int(hpo['stride']), int(hpo['stride']))))
  
  # Delete neuron randomly while training
  # Regularization technique to avoid overfitting
  model.add(Dropout(hpo['dropout_l1']))
  
  # Flatten layer 
  model.add(Flatten())
  
  # Fully connected Layer
  model.add(Dense(128, activation='relu'))
  
  # Delete neuron randomly while training 
  # Regularization technique to avoid overfitting
  model.add(Dropout(hpo['dropout_l2']))
  
  # Apply Softmax
  model.add(Dense(num_classes, activation='softmax'))

  # Select SGD Algorithm
  optimizer_call = getattr(keras.optimizers, hpo['optimizer'])
  optimizer = optimizer_call(math.pow(10, hpo['learning_rate']))
  return model

# COMMAND ----------

# MAGIC %md ### Train and Test model, log parameters & Results to MLFlow

# COMMAND ----------

def runCNN(hpo):

  model = create_model(hpo)
  
  # MLflow Tracking
  with mlflow.start_run(run_name=RUN_NAME) as run:
    # compile model
    model.compile(loss=getattr(keras.losses, hpo['loss_function']),
                optimizer=optimizer,
                metrics=['accuracy'])
    
    # Fit model + track train time
    start_train_time = time.time()
    
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=single_node_epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    
    complete_train_time = time.time()

    # Evaluate our model + track evaluation time
    score = model.evaluate(x_test, y_test, verbose=0)
    
    complete_eval_time = time.time()
    
    # Calculate Training time and Evaluation time
    total_train_time = complete_train_time - start_train_time
    total_eval_time = complete_eval_time - complete_train_time
    
    # Log MLflow Parameters
    mlflow.log_param("mode", "single_node")
    mlflow.log_param("kernel", hpo['kernel'])
    mlflow.log_param("conv", hpo['conv'])
    mlflow.log_param("stride", hpo['stride'])
    mlflow.log_param("dropout_l1", hpo['dropout_l1'])
    mlflow.log_param("dropout_l2", hpo['dropout_l2'])
    mlflow.log_param("learning_rate", hpo['learning_rate'])
    mlflow.log_param("SGD_flavor", hpo['optimizer'])
    mlflow.log_param("Loss Function", hpo['loss_function'])
    
    # Log MLflow Metrics
    mlflow.log_metric("Test Loss", score[0])
    mlflow.log_metric("Test Accuracy", score[1])
    mlflow.log_metric("Train Duration", total_train_time)
    mlflow.log_metric("Test Duration", total_eval_time)
         
    # Log Model
    mlflow.keras.log_model(model, "models")
    return {'loss': -score[1], 'status': STATUS_OK}

  # Close TF session
  sess.close()
  
  # Complete MLflow Run
  mlflow.end_run()  

# COMMAND ----------

# MAGIC %md ### Setup hyperparameter space and training config, and then invoke model training via HyperOpt

# COMMAND ----------

batch_size = 128
single_node_epochs = 36

space = {'kernel': hp.quniform('kernel', 2,4,1),
         'conv': hp.quniform('conv', 2,4,1),
         'stride': hp.quniform('stride', 2,4,1),
         'dropout_l1': hp.uniform('dropout_l1', .25,.75),
         'dropout_l2': hp.uniform('dropout_l2', .25,.75),
         'learning_rate': hp.uniform('learning_rate', -10,1),
         'optimizer': hp.choice('optimizer', ['adadelta','adam','rmsprop']),
         'loss_function': hp.choice('loss_function', ['categorical_crossentropy'])
        }

fmin(runCNN, space, algo=tpe.suggest, max_evals=105)

# COMMAND ----------

# MAGIC %md ### Analyzing results in MLFlow
# MAGIC 
# MAGIC 1) Go to mlflow, and the project <EXPERIMENT_NAME> as defined in cell 5  
# MAGIC 
# MAGIC 2) Explain UI (user, notebook, parameters, metrics, etc.)  
# MAGIC 
# MAGIC 3) Select-all and hit compare  
# MAGIC 
# MAGIC 4) Look at distribution of accuracy and test_duration results.  Highlight min/max/range of test_duration results.  
# MAGIC 
# MAGIC 5) Return to tabular format  
# MAGIC 
# MAGIC 6) Split out and sort by accuracy.  Address anything common (or explictly distinct) regarding top 3-4 results.  
# MAGIC   * Which results should you carry forward to test against holdout?  
# MAGIC   * Looking at the loss function, which might we assume will generalize the best?  
# MAGIC   
# MAGIC 7) Split and sort by test_duration  
# MAGIC   * Explain test_duration as proxy for inference time (particularly important for realtime aplications)  
# MAGIC   * Scroll down to run with accuracy close to "best" accuracy  
# MAGIC   
# MAGIC 8) Highlight that all of this will help them not only make better choices about what to deploy, but also informs what they may want to try next, how they might re-scope the hyperparameter space for future experiments to decrease training time and/or improve results, etc.  

# COMMAND ----------

