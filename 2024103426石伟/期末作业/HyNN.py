# For example, this script execute the HyNN in ULA-64 antennas dataset, in slurm. 
# Executed in slurm with the following script:
"""
#!/bin/bash
#SBATCH --job-name=HyNN
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=300GB
#SBATCH --output=hynn/slurm-%A_%a.out
#SBATCH --error=hynn/slurm-%A_%a.err

cd $SLURM_SUBMIT_DIR

source $HOME/.bashrc
micromamba activate gpu_env

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

DATASET_PATH="/data/datasets/ULA_lab_LoS_64.csv" # Change the dataset to the one you want to use (ULA, URA, DIS setup; 8, 16, 32, 64 antennas)
SETUP = "ULA" # Change the setup to the one you want to use (ULA, URA, DIS)
ANTENNAS = "64" # Change the number of antennas to the one you want to use (8, 16, 32, 64)

# Execute with: sbatch HyNN.sh
python HyNN.py $DATASET_PATH $SETUP $ANTENNAS
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import time
import gc

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Multiply, Add, Concatenate, Dense, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Activation,MaxPooling2D
from keras.layers import MultiHeadAttention, LayerNormalization, Lambda

from TINTOlib.tinto import TINTO

df = pd.read_csv("./data/datasets/ULA_lab_LoS_64.csv")
setup = "ULA" # In string format
num_antennas = "64" # In string format

#Select the model and the parameters
problem_type = "regression"
pixel = 35
image_model = TINTO(problem= problem_type,pixels=pixel,blur=True)

images_folder = "/images_" + num_antennas + "antenas_" + setup # "/images_64antenas_ULA"
results_folder = "/results/"



# * NORMALIZE DATASET

# Select all the attributes to normalize
columns_to_normalize = df.columns[:-2]

# Normalize between 0 and 1
df_normalized = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

# Combine the attributes and the label
# df_normalized = pd.concat([df_normalized, df[df.columns[-1]]], axis=1)
df_normalized = pd.concat([df_normalized, df[df.columns[-2]], df[df.columns[-1]]], axis=1)


#Generate the images with TINTO

if not os.path.exists(images_folder):
   print("generating images...")
   image_model.generateImages(df.iloc[:,:-1], images_folder)
   print("Images generated")
   
   # Save the TINTO model
   pickle.dump(image_model, open(images_folder + "/image_model.pkl", "wb"))
   

if not os.path.exists(images_folder+results_folder):
   os.makedirs(images_folder+results_folder)
   
img_paths = os.path.join(images_folder,problem_type+".csv")


imgs = pd.read_csv(img_paths) 

imgs["images"]= images_folder + "/" + imgs["images"] 

imgs["images"] = imgs["images"].str.replace("\\","/")


combined_dataset_x = pd.concat([imgs,df_normalized.iloc[:,:-1]],axis=1)
combined_dataset_y = pd.concat([imgs,pd.concat([df_normalized.iloc[:,:-2], df_normalized.iloc[:,-1:]],axis=1)],axis=1)  

#df_x = combined_dataset.drop("homa_b",axis=1).drop("values",axis=1)
df_x = combined_dataset_x.drop("PositionX",axis=1).drop("values",axis=1)
df_y_for_x = combined_dataset_x["values"]
df_y_for_y = combined_dataset_y["PositionY"]

np.random.seed(64)
df_x = df_x.sample(frac=1).reset_index(drop=True)

np.random.seed(64)
df_y_for_x = df_y_for_x.sample(frac=1).reset_index(drop=True)

np.random.seed(64)
df_y_for_y = df_y_for_y.sample(frac=1).reset_index(drop=True)


# Training size
trainings_size = 0.85                     # 85% training set
validation_size = 0.1                     # 10% validation set
test_size = 0.05                         # 5% test set

import cv2

# Split the dataset into training, validation and test sets
X_train = df_x.iloc[:int(trainings_size*len(df_x))]
y_train_x = df_y_for_x.iloc[:int(trainings_size*len(df_y_for_x))]
y_train_y = df_y_for_y.iloc[:int(trainings_size*len(df_y_for_y))]

X_val = df_x.iloc[int(trainings_size*len(df_x)):int((trainings_size+validation_size)*len(df_x))]
y_val_x = df_y_for_x.iloc[int(trainings_size*len(df_y_for_x)):int((trainings_size+validation_size)*len(df_y_for_x))]
y_val_y = df_y_for_y.iloc[int(trainings_size*len(df_y_for_y)):int((trainings_size+validation_size)*len(df_y_for_y))]

X_test = df_x.iloc[-int(test_size*len(df_x)):]
y_test_x = df_y_for_x.iloc[-int(test_size*len(df_y_for_x)):]
y_test_y = df_y_for_y.iloc[-int(test_size*len(df_y_for_y)):]

X_train_num = X_train.drop("images",axis=1)
X_val_num = X_val.drop("images",axis=1)
X_test_num = X_test.drop("images",axis=1)

# For 3 RGB channels
X_train_img = np.array([cv2.resize(cv2.imread(img),(pixel,pixel)) for img in X_train["images"]])
X_val_img = np.array([cv2.resize(cv2.imread(img),(pixel,pixel)) for img in X_val["images"]])
X_test_img = np.array([cv2.resize(cv2.imread(img),(pixel,pixel)) for img in X_test["images"]])
   

validation_x = y_val_x
test_x = y_test_x
validation_y = y_val_y
test_y = y_test_y

shape = len(X_train_num.columns)


from keras.layers import AveragePooling2D, Concatenate

dropout = 0.3

filters_ffnn = [1024,512,256,128,64,32,16]

ff_inputs = Input(shape = (shape,))

# * START BRANCH 1
mlp_1 = Dense(1024, activation='relu')(ff_inputs)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(512, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(256, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(128, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(64, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(32, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(16, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

# * START BRANCH 2
mlp_2 = Dense(1024, activation='relu')(ff_inputs)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(512, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(256, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(128, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(64, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(32, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(16, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

merged_tabular = Concatenate(axis=1)([mlp_1, mlp_2])

ff_model = Model(inputs = ff_inputs, outputs = merged_tabular)

# * CNN

#input
input_shape = Input(shape=(pixel, pixel, 3))

#Start branch 1
tower_1 = Conv2D(16, (3,3), activation='relu',padding="same")(input_shape)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)

tower_1 = Conv2D(32, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)

tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)

tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)
#End branch 1

#Start branch 2
tower_2 = Conv2D(16, (5,5), activation='relu',padding="same")(input_shape)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)

tower_2 = Conv2D(32, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)

tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)

tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)
#End branch 2

#Concatenation of the 2 branches
merged = Concatenate(axis=1)([tower_1, tower_2])

#Flattening
merged = Flatten()(merged)

#Additional layers
out = Dense(512, activation='relu')(merged)
out = Dropout(dropout)(merged)
out = Dense(256, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(128, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(64, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(32, activation='sigmoid')(out)
out = Dropout(dropout)(out)



cnn_model = Model(input_shape, out)

from keras.layers import MultiHeadAttention, LayerNormalization, Lambda

# Concatenate the outputs of both branches to form the input of the Transformer
transformer_input = Concatenate(axis=1)([ff_model.output, cnn_model.output])

# Add a dimension so that MultiHeadAttention treats it as a sequence
transformer_input = Lambda(lambda x: tf.expand_dims(x, axis=1))(transformer_input)

# Transformer Block: Multi-Head Attention
attention_output = MultiHeadAttention(num_heads=12, key_dim=1024)(transformer_input, transformer_input)

# Residual Connection + Layer Normalization
attention_output = Add()([transformer_input, attention_output])
attention_output = LayerNormalization()(attention_output)

# Feed-Forward inside the Transformer Block
transformer_ff = Dense(2048, activation="relu")(attention_output)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(1024, activation="relu")(transformer_ff)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(512, activation="relu")(transformer_ff)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(256, activation="relu")(transformer_ff)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(128, activation="relu")(transformer_ff)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(64, activation="relu")(transformer_ff)

# Residual Connection + Layer Normalization
transformer_output = Add()([attention_output, transformer_ff])
transformer_output = LayerNormalization()(transformer_output)

# Flatten the output of the Transformer
flattened_output = Flatten()(transformer_output)

# Final layers after the fusion
x = Dense(64, activation="relu")(flattened_output)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

# Final Model
modelX = Model(inputs=[ff_model.input, cnn_model.input], outputs=x)


from tensorflow_addons.metrics import RSquare

METRICS = [
   tf.keras.metrics.MeanSquaredError(name = 'mse'),
   tf.keras.metrics.MeanAbsoluteError(name = 'mae'),
   #tf.keras.metrics.R2Score(name = 'r2'),
   RSquare(name='r2_score'),
   tf.keras.metrics.RootMeanSquaredError(name = 'rmse')
]

opt = Adam()
modelX.compile(
   loss="mse",
   optimizer=opt,
   metrics = METRICS
)

# Define EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


t0 = time.time()

model_history=modelX.fit(
   x=[X_train_num, X_train_img], y=y_train_x,
   validation_data=([X_val_num, X_val_img], y_val_x),
   epochs=200,
   batch_size=32,
   verbose=2,
   callbacks=[early_stopping]
   #verbose=2
   #steps_per_epoch = X_train_num.shape[0]//batch_size,
   #validation_steps = X_train_num.shape[0]//batch_size,
)
print("TRAIN TIME: ", time.time()-t0)


modelX.save(images_folder+results_folder+'/modelX.h5')


# RESULTS

plt.plot(model_history.history['loss'], color = 'red', label = 'loss')
plt.plot(model_history.history['val_loss'], color = 'green', label = 'val loss')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'loss_graphX.png')
plt.clf()


plt.plot(model_history.history['mse'], color = 'red', label = 'mse')
plt.plot(model_history.history['val_mse'], color = 'green', label = 'val mse')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'mse_graphX.png')
plt.clf()


plt.plot(model_history.history['mae'], color = 'red', label = 'mae')
plt.plot(model_history.history['val_mae'], color = 'green', label = 'val mae')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'mae_graphX.png')
plt.clf()

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
#                              MODEL FOR Y
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////


dropout = 0.3

filters_ffnn = [1024,512,256,128,64,32,16]

ff_inputs = Input(shape = (shape,))


# * START BRANCH 1
mlp_1 = Dense(1024, activation='relu')(ff_inputs)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(512, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(256, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(128, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(64, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(32, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(16, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

# * START BRANCH 2
mlp_2 = Dense(1024, activation='relu')(ff_inputs)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(512, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(256, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(128, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(64, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(32, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(16, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

merged_tabular = Concatenate(axis=1)([mlp_1, mlp_2])

ff_model = Model(inputs = ff_inputs, outputs = merged_tabular)


#Input
input_shape = Input(shape=(pixel, pixel, 3))

#Start branch 1
tower_1 = Conv2D(16, (3,3), activation='relu',padding="same")(input_shape)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)

tower_1 = Conv2D(32, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)

tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)

tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)
#End branch 1

#Start branch 2
tower_2 = Conv2D(16, (5,5), activation='relu',padding="same")(input_shape)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)

tower_2 = Conv2D(32, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)

tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)

tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)
#End branch 2

#Concatenation of the 2 branches
merged = Concatenate(axis=1)([tower_1, tower_2])

#Flattening
merged = Flatten()(merged)

#Additional layers
out = Dense(512, activation='relu')(merged)
out = Dropout(dropout)(merged)
out = Dense(256, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(128, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(64, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(32, activation='sigmoid')(out)
out = Dropout(dropout)(out)


#out = Dense(n_class, activation='softmax')(out)
cnn_model = Model(input_shape, out)

from keras.layers import MultiHeadAttention, LayerNormalization, Lambda

# Concatenate the outputs of both branches to form the input of the Transformer
transformer_input = Concatenate(axis=1)([ff_model.output, cnn_model.output])

# Add a dimension so that MultiHeadAttention treats it as a sequence
transformer_input = Lambda(lambda x: tf.expand_dims(x, axis=1))(transformer_input)

# Transformer Block: Multi-Head Attention
attention_output = MultiHeadAttention(num_heads=12, key_dim=1024)(transformer_input, transformer_input)

# Residual Connection + Layer Normalization
attention_output = Add()([transformer_input, attention_output])
attention_output = LayerNormalization()(attention_output)

# Feed-Forward inside the Transformer Block
transformer_ff = Dense(2048, activation="relu")(attention_output)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(1024, activation="relu")(transformer_ff)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(512, activation="relu")(transformer_ff)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(256, activation="relu")(transformer_ff)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(128, activation="relu")(transformer_ff)
transformer_ff = Dropout(dropout)(transformer_ff)
transformer_ff = Dense(64, activation="relu")(transformer_ff)

# Residual Connection + Layer Normalization
transformer_output = Add()([attention_output, transformer_ff])
transformer_output = LayerNormalization()(transformer_output)

# Flatten the output of the Transformer
flattened_output = Flatten()(transformer_output)

# Final layers after the fusion
x = Dense(64, activation="relu")(flattened_output)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

# Final model
modelY = Model(inputs=[ff_model.input, cnn_model.input], outputs=x)


METRICS = [
   tf.keras.metrics.MeanSquaredError(name = 'mse'),
   tf.keras.metrics.MeanAbsoluteError(name = 'mae'),
   #tf.keras.metrics.R2Score(name='r2_score'),
   RSquare(name='r2_score'),
   tf.keras.metrics.RootMeanSquaredError(name='rmse')
]

opt = Adam(learning_rate=1e-4)
modelY.compile(
   loss="mse",
   optimizer=opt,
   metrics = METRICS
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

t0 = time.time()

model_history=modelY.fit(
   x=[X_train_num, X_train_img], y=y_train_y,
   validation_data=([X_val_num, X_val_img], y_val_y),
   epochs=200,
   batch_size=32,
   verbose=2,
   callbacks=[early_stopping]
   #verbose=2
   #steps_per_epoch = X_train_num.shape[0]//batch_size,
   #validation_steps = X_train_num.shape[0]//batch_size,
)
print("TRAIN TIME: ", time.time()-t0)

modelY.save(images_folder+results_folder+'/modelY.h5')

# RESULTS

plt.plot(model_history.history['loss'], color = 'red', label = 'loss')
plt.plot(model_history.history['val_loss'], color = 'green', label = 'val loss')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'loss_graphY.png')
plt.clf()


plt.plot(model_history.history['mse'], color = 'red', label = 'mse')
plt.plot(model_history.history['val_mse'], color = 'green', label = 'val mse')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'mse_graphY.png')
plt.clf()


plt.plot(model_history.history['mae'], color = 'red', label = 'mae')
plt.plot(model_history.history['val_mae'], color = 'green', label = 'val mae')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'mae_graphY.png')
plt.clf()


def true_dist(y_pred, y_true):
   return np.mean(np.sqrt(
       np.square(np.abs(y_pred[:,0] - y_true[:,0]))
       + np.square(np.abs(y_pred[:,1] - y_true[:,1]))
       ))

# VALIDATION RESULTS
folder = images_folder+results_folder+"/predictions/validation/"
if not os.path.exists(folder):
   os.makedirs(folder)

start_time = time.time()
predX_val = modelX.predict([X_val_num,X_val_img])
print("PREDICTION TIME OF X (VALIDATION): ", time.time()-start_time)

Start_time = time.time()
predY_val = modelY.predict([X_val_num,X_val_img])
print("PREDICTION TIME OF Y (VALIDATION): ", time.time()-start_time)

validation_preds = pd.DataFrame()
validation_preds["realX"] = validation_x 
validation_preds["realY"] = validation_y 

validation_preds["predX"] = predX_val
validation_preds["predY"] = predY_val

validation_preds.to_csv(folder+'preds_val.csv', index=False)

error_valid = true_dist(validation_preds[["predX", "predY"]].to_numpy(), validation_preds[["realX", "realY"]].to_numpy())
print(error_valid)



# RESULTS FOR TEST

start_time = time.time()
predX_test = modelX.predict([X_test_num,X_test_img])
print("PREDICTION TIME OF X (TEST): ", time.time()-start_time)

start_time = time.time()
predY_test = modelY.predict([X_test_num,X_test_img])
print("PREDICTION TIME OF Y (TEST): ", time.time()-start_time)

folder = images_folder+results_folder+"/prediction/test/"
if not os.path.exists(folder):
   os.makedirs(folder)

test_preds = pd.DataFrame()
test_preds["realX"] = test_x
test_preds["realY"] = test_y

test_preds["predX"] = predX_test
test_preds["predY"] = predY_test

test_preds.to_csv(folder+'preds_test.csv', index=False)

error_test = true_dist(test_preds[["predX", "predY"]].to_numpy(), test_preds[["realX", "realY"]].to_numpy())

print(error_test)


# TEST X的评估指标
mae_x = mean_absolute_error(y_test_x, predX_test)
mse_x = mean_squared_error(y_test_x, predX_test)
rmse_x = np.sqrt(mse_x)  # 修改这里
r2_x = r2_score(y_test_x, predX_test)

# TEST Y的评估指标
mae_y = mean_absolute_error(y_test_y, predY_test)
mse_y = mean_squared_error(y_test_y, predY_test)
rmse_y = np.sqrt(mse_y)  # 修改这里
r2_y = r2_score(y_test_y, predY_test)

# Print the evaluation metrics
print("TEST X:")
print("Mean Absolute Error:", mae_x)
print("Mean Squared Error:", mse_x)
print("Root Mean Squared Error:", rmse_x)
print("R2 Score:", r2_x)
print()
print("TEST Y:")    
print("Mean Absolute Error:", mae_y)
print("Mean Squared Error:", mse_y)
print("Root Mean Squared Error:", rmse_y)
print("R2 Score:", r2_y)

# Save evaluation metrics to a text file
results_filename = 'evaluation_results.txt'
with open(results_filename, 'w') as results_file:
   results_file.write("Evaluation Metrics FOR X:\n")
   results_file.write(f"Mean Absolute Error: {mae_x}\n")
   results_file.write(f"Mean Squared Error: {mse_x}\n")
   results_file.write(f"Root Mean Squared Error: {rmse_x}\n")
   results_file.write(f"R2 Score: {r2_x}\n")
   results_file.write("\n")
   results_file.write("Evaluation Metrics FOR Y:\n")
   results_file.write(f"Mean Absolute Error: {mae_y}\n")
   results_file.write(f"Mean Squared Error: {mse_y}\n")
   results_file.write(f"Root Mean Squared Error: {rmse_y}\n")
   results_file.write(f"R2 Score: {r2_y}\n")
   results_file.write("\n")
   results_file.write(f"Validation Mean Error: {error_valid}\n")
   results_file.write(f"Test Mean Error: {error_test}\n")

# 在代码末尾添加优化后的权重保存逻辑
import os
import tensorflow as tf
import numpy as np

# 创建权重保存目录（如果不存在）
weights_dir = os.path.join(images_folder, results_folder)
os.makedirs(weights_dir, exist_ok=True)


# 1. 保存为TF Lite格式（大幅减小文件大小）
def save_as_tflite(model, model_name):
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   # 使用默认优化
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   # 尝试量化到FP16（进一步减小大小）
   converter.target_spec.supported_types = [tf.float16]
   tflite_model = converter.convert()

   tflite_path = os.path.join(weights_dir, f'{model_name}.tflite')
   with open(tflite_path, 'wb') as f:
      f.write(tflite_model)

   print(f"{model_name} saved as TFLite: {tflite_path}")
   print(f"File size: {len(tflite_model) / 1024 / 1024:.2f} MB")
   return tflite_path


# 保存模型为TFLite格式
save_as_tflite(modelX, 'modelX')
save_as_tflite(modelY, 'modelY')


# 2. 仅保存权重（使用压缩）
def save_compressed_weights(model, model_name):
   weights = model.get_weights()

   # 创建压缩的权重文件
   weights_path = os.path.join(weights_dir, f'{model_name}_weights.npz')
   np.savez_compressed(weights_path, *weights)

   print(f"{model_name} weights saved (compressed): {weights_path}")
   print(f"File size: {os.path.getsize(weights_path) / 1024 / 1024:.2f} MB")
   return weights_path


# 保存压缩后的权重
save_compressed_weights(modelX, 'modelX')
save_compressed_weights(modelY, 'modelY')


# 3. 保存模型架构（可选，用于后续重建）
def save_model_architecture(model, model_name):
   # 保存为JSON
   json_path = os.path.join(weights_dir, f'{model_name}_architecture.json')
   with open(json_path, 'w') as json_file:
      json_file.write(model.to_json())

   print(f"{model_name} architecture saved: {json_path}")
   return json_path


save_model_architecture(modelX, 'modelX')
save_model_architecture(modelY, 'modelY')
