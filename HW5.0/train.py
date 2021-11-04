import pandas as pd 
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Embedding
from keras.layers import LSTM,SimpleRNN
from keras.optimizers import RMSprop,Adam,SGD
from keras import regularizers
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import label_binarize


from keras.callbacks import CSVLogger

csv_logger = CSVLogger('training.log', separator=',', append=False)


df=pd.read_csv("cleaned_texts.csv")
print(df.shape)
n_classes=3
model_type='CNN'
texts=df['text'].tolist()
labels=df['label'].tolist()
max_features = 10000
max_len = 100
training_samples = 600
validation_samples = 100
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=max_len)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
labels=label_binarize(labels, classes=[0, 1, 2])
print(data.shape)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

glove_dir = '/home/jay/Downloads/glove.6B'


embedding_dim = 100



# # Training the same model without pretrained word embeddings


if model_type=='CNN':
	# model = Sequential()
	# model.add(layers.Embedding(max_features, 32, input_length=maxlen))
	# model.add(layers.Conv1D(32, 7, activation='relu'))
	# model.add(layers.MaxPooling1D(5))
	# model.add(layers.Conv1D(32, 7, activation='relu'))
	# model.add(layers.GlobalMaxPooling1D())
	# model.add(layers.Dense(1,activation='sigmoid'))
	# model.summary()
	model = Sequential()
	model.add(layers.Embedding(max_features, 32, input_length=max_len))
	model.add(layers.Conv1D(32, 7, activation='relu',kernel_regularizer=regularizers.l2(0.02)))
	model.add(layers.Dropout(0.2))
	model.add(layers.MaxPooling1D(5))
	model.add(layers.Conv1D(32, 7, activation='relu'))
	model.add(layers.Dropout(0.2))
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(3,activation='softmax'))
	model.compile(optimizer=RMSprop(lr=1e-2),
	loss='categorical_crossentropy',
	metrics=['accuracy'])
elif model_type=='simple_RNN':
	model = Sequential()
	model.add(Embedding(max_features, 32))
	model.add(SimpleRNN(32))
	model.add(Dense(3, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy',auroc])
elif model_type=="LSTM":

	model = Sequential()
	model.add(Embedding(max_features, 32))
	model.add(LSTM(32))
	model.add(Dense(3, activation='softmax'))
	model.compile(optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy',auroc])


# elif model_type=="GRU":
# 	model = Sequential()
# 	model.add(Embedding(max_features, 32))
# 	model.add(layers.GRU(32,
# 	dropout=0.2,
# 	recurrent_dropout=0.2,
# 	input_shape=(None, x_train.shape[-1])))
# 	model.add(layers.Dense(1))
# 	model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')
# 	history = model.fit_generator(x_train,
# 	steps_per_epoch=500,
# 	epochs=40,
# 	validation_data=val_gen,
# 	validation_steps=val_steps)
model.save('pre_trained_model.h5')
history = model.fit(x_train, y_train,
epochs=20,
batch_size=32,
validation_split=0.2,callbacks=[csv_logger])

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss_plot.png')
from sklearn.metrics import roc_curve
# make a prediction
y_score = model.predict(x_val)
# #Print Area Under Curve
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
labels_map={0:"Fiander's Widow", 1:"Monday or Tuesday",2:"The Castle of Otranto"}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print('AUC score for title %s is %f' % (labels_map[i],roc_auc[i]))
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('ROC_AUC.png')
# ### RNN
# # Simple RNN
# from keras.layers import Dense
# model = Sequential()
# model.add(Embedding(max_features, 32))
# model.add(SimpleRNN(32))
# model.add(Dense(1, activation='softmax'))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# history = model.fit(input_train, y_train,
# epochs=10,
# batch_size=128,
# validation_split=0.2)


# #LSTM

# from keras.layers import LSTM
# model = Sequential()
# model.add(Embedding(max_features, 32))
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
# loss='binary_crossentropy',
# metrics=['acc'])
# history = model.fit(input_train, y_train,
# epochs=10,
# batch_size=128,
# validation_split=0.2)


# #Training and evaluating a dropout-regularized GRU-based model
# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
# model = Sequential()
# model.add(layers.GRU(32,
# dropout=0.2,
# recurrent_dropout=0.2,
# input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
# steps_per_epoch=500,
# epochs=40,
# validation_data=val_gen,
# validation_steps=val_steps)