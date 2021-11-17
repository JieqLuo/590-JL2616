import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from keras import models
from keras import layers
import keras
from sklearn import metrics
from keras.callbacks import CSVLogger
import logging
import os



csv_logger = CSVLogger('training_6_1.log', separator=',', append=False)


#USER PARAM
model_run=True

#GET DATASET
from keras.datasets import mnist,fashion_mnist
(X, Y), (test_images, test_labels) = mnist.load_data()
(X_fashion, Y_fashion), (fashion_test_images, fashion_test_labels) = fashion_mnist.load_data()



#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(60000,28*28); 

X_fashion=X_fashion/np.max(X_fashion) 
X_fashion=X_fashion.reshape(60000,28*28); 

test_images=test_images/np.max(test_images) 
test_images=test_images.reshape(10000,28*28); 

EPOCHS          =   10
BATCH_SIZE      =   1024
N_channels=1; PIX=28

if model_run==True:
	#MODEL
	n_bottleneck=100
	NH=200
	#DEEPER
	input_img = keras.Input(shape=(PIX*PIX))
	encoded=layers.Dense(NH, activation='relu')(input_img)
	bottleneck=layers.Dense(n_bottleneck, activation='relu')(encoded)
	decoded=layers.Dense(NH, activation='relu')(bottleneck)
	decoded=layers.Dense(NH, activation='relu')(decoded)
	decoded=layers.Dense(28*28,  activation='sigmoid')(decoded)


	#COMPILE AND FIT
	autoencoder = keras.Model(input_img, decoded)

	autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error',	metrics=['accuracy']);
	autoencoder.summary()

	#TRAIN
	history = autoencoder.fit(X, X,
	                epochs=EPOCHS,
	                batch_size=BATCH_SIZE,
	                validation_split=0.2,callbacks=[csv_logger])


	autoencoder.save('hw6_1_AE_model.h5')
	#HISTORY PLOT
	epochs = range(1, len(history.history['loss']) + 1)
	plt.figure()
	plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
	plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig('loss_plot_6_1.png')

	plt.figure()
	plt.plot(epochs, history.history['accuracy'], 'bo', label='Training acc')
	plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.savefig('accuracy_plot_6_1.png')

else:

	autoencoder=keras.models.load_model("hw6_1_AE_model.h5")
	# results = autoencoder.evaluate(fashion_test_images, fashion_test_labels, batch_size=128)
	# print(results)






# Anomaly detection 
logFile = 'training_6_1.log'
logging.basicConfig( filename = logFile,filemode = 'a',level=logging.INFO)

##	MAKE PREDICTIONS FOR fashion DATA
decoded_imgs_train = autoencoder.predict(X,batch_size=BATCH_SIZE)
decoded_imgs_test = autoencoder.predict(test_images,batch_size=BATCH_SIZE)
decoded_imgs_fashion = autoencoder.predict(X_fashion,batch_size=BATCH_SIZE)

mse_minist_train = metrics.mean_squared_error(X, decoded_imgs_train)
print(mse_minist_train)


anomaly_index=[]
for i in range(test_images.shape[0]):
	mse = metrics.mean_squared_error(test_images[i], decoded_imgs_test[i])
	if mse_minist_train*1<=mse:
		anomaly_index.append(i)
fraction_anomaly=len(anomaly_index)/test_images.shape[0]
text="the anomaly fraction for minist test dataset is: %f" % (fraction_anomaly)
print(text)
logging.info(text)



anomaly_index=[]
for i in range(X_fashion.shape[0]):
	mse = metrics.mean_squared_error(X_fashion[i], decoded_imgs_fashion[i])
	if mse_minist_train*1<=mse:
		anomaly_index.append(i)
fraction_anomaly=len(anomaly_index)/X_fashion.shape[0]
print(fraction_anomaly)
text="the anomaly fraction for fashion minist test dataset is: %f" % (fraction_anomaly)
print(text)
logging.info(text)
#VISUALIZE THE RESULTS


n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(test_images[i].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs_test[i].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('original_and_decoded_minist_image_6_1.png')

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_fashion[anomaly_index[i]].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs_fashion[anomaly_index[i]].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('original_and_decoded_fashion_image_6_1.png')