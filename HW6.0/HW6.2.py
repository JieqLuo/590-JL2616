import keras
from keras import layers
import matplotlib.pyplot as plt

from keras.datasets import mnist,cifar10,fashion_mnist
import numpy as np
from sklearn import metrics
from keras.callbacks import CSVLogger
import logging
import os



csv_logger = CSVLogger('training_6_2.log', separator=',', append=False)

#USER PARAM
INJECT_NOISE    =   False
EPOCHS          =   20
NKEEP           =   10000       #DOWNSIZE DATASET
BATCH_SIZE      =   500
DATA            =   "MNIST"
model_run=True

(X_fashion, Y_fashion), (fashion_test_images, fashion_test_labels) = fashion_mnist.load_data()

#GET DATA
if(DATA=="MNIST"):
    (x_train, _), (x_test, _) = mnist.load_data()
    N_channels=1; PIX=28

if(DATA=="CIFAR"):
    (x_train, _), (x_test, _) = cifar10.load_data()
    N_channels=3; PIX=32
    EPOCHS=100 #OVERWRITE

#NORMALIZE AND RESHAPE
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

X_fashion=X_fashion.astype('float32') / 255
Y_fashion=Y_fashion.astype('float32') / 255


#DOWNSIZE TO RUN FASTER AND DEBUG
print("BEFORE",x_train.shape)
x_train=x_train[0:NKEEP]
x_test=x_test[0:NKEEP]
print("AFTER",x_train.shape)

#ADD NOISE IF DENOISING
if(INJECT_NOISE):
    EPOCHS=2*EPOCHS
    #GENERATE NOISE
    noise_factor = 0.5
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_train=x_train+noise
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    x_test=x_test+noise

    #CLIP ANY PIXELS OUTSIDE 0-1 RANGE
    x_train = np.clip(x_train, 0., 1.)
    x_test = np.clip(x_test, 0., 1.)

#BUILD CNN-AE MODEL

if model_run==True:
    if(DATA=="MNIST"):
        input_img = keras.Input(shape=(PIX, PIX,N_channels))

        # #ENCODER
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        # # AT THIS POINT THE REPRESENTATION IS (4, 4, 8) I.E. 128-DIMENSIONAL
     
        # #DECODER
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)


    if(DATA=="CIFAR"):
        input_img = keras.Input(shape=(PIX, PIX, N_channels))

        #ENCODER
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPoolinNKEEPg2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        #DECODER
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)



    #COMPILE
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy']);
    autoencoder.summary()

    #TRAIN
    history = autoencoder.fit(x_train, x_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(x_test, x_test),callbacks=[csv_logger])

    autoencoder.save('hw6_2_CNN_AE_model.h5')
    #HISTORY PLOT
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('loss_plot_6_2.png')

    plt.figure()
    plt.plot(epochs, history.history['accuracy'], 'bo', label='Training acc')
    plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('accuracy_plot_6_2.png')
else:
    autoencoder=keras.models.load_model("hw6_1_AE_model.h5")
    # results = autoencoder.evaluate(fashion_test_images, fashion_test_labels, batch_size=128)
    # print(results)




# Anomaly detection 
logFile = 'training_6_2.log'
logging.basicConfig( filename = logFile,filemode = 'a',level=logging.INFO)

##  MAKE PREDICTIONS FOR DATA
decoded_imgs_train = autoencoder.predict(x_train,batch_size=BATCH_SIZE)
decoded_imgs_test = autoencoder.predict(x_test,batch_size=BATCH_SIZE)
decoded_imgs_fashion = autoencoder.predict(X_fashion,batch_size=BATCH_SIZE)


print(x_train.shape)
print(x_test.shape)
print(X_fashion.shape)
print('here')
x_train=x_train.reshape(NKEEP,28*28)
decoded_imgs_train=decoded_imgs_train.reshape(NKEEP,28*28)
x_test=x_test.reshape(NKEEP,28*28)
decoded_imgs_test=decoded_imgs_test.reshape(NKEEP,28*28)

X_fashion=X_fashion.reshape(60000,28*28)
decoded_imgs_fashion=decoded_imgs_fashion.reshape(60000,28*28)



mse_minist_train = metrics.mean_squared_error(x_train, decoded_imgs_train)
print(mse_minist_train)


anomaly_index=[]
for i in range(x_test.shape[0]):
    mse = metrics.mean_squared_error(x_test[i], decoded_imgs_test[i])
    if mse_minist_train*1<=mse:
        anomaly_index.append(i)
fraction_anomaly=len(anomaly_index)/x_test.shape[0]
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


x_train=x_train.reshape(NKEEP,28,28,1)
decoded_imgs_train=decoded_imgs_train.reshape(NKEEP,28,28,1)
x_test=x_test.reshape(NKEEP,28,28,1)
decoded_imgs_test=decoded_imgs_test.reshape(NKEEP,28,28,1)

X_fashion=X_fashion.reshape(60000,28,28,1)
decoded_imgs_fashion=decoded_imgs_fashion.reshape(60000,28,28,1)


n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(PIX,PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs_test[i].reshape(PIX,PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('original_and_decoded_minist_image_6_2.png')

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_fashion[anomaly_index[i]].reshape(PIX,PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs_fashion[anomaly_index[i]].reshape(PIX,PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('original_and_decoded_fashion_image_6_2.png')
