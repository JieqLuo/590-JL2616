import pandas as pd 
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM,SimpleRNN
from keras.optimizers import RMSprop
import numpy as np

df=pd.read_csv("cleaned_texts.csv")
print(df.shape)

model_type='CNN'
texts=df['text'].tolist()
labels=df['label'].tolist()
max_features = 10000
maxlen = 100
training_samples = 500
validation_samples = 100
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

glove_dir = '/home/jay/Downloads/glove.6B'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
	if i < max_words:
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector



# # Training the same model without pretrained word embeddings


if model_type=='CNN':
	model = Sequential()
	model.add(layers.Embedding(max_features, 128, input_length=maxlen))
	model.add(layers.Conv1D(32, 7, activation='relu'))
	model.add(layers.MaxPooling1D(5))
	model.add(layers.Conv1D(32, 7, activation='relu'))
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(1))
	model.summary()
	model.compile(optimizer=RMSprop(lr=1e-4),
	loss='categorical_crossentropy',
	metrics=['accuracy'])
	history = model.fit(x_train, y_train,
	epochs=10,
	batch_size=128,
	validation_split=0.2)
elif model_type=='simple_RNN':
	model = Sequential()
	model.add(Embedding(max_features, 128))
	model.add(SimpleRNN(128))
	model.add(Dense(1, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit(x_train, y_train,
	epochs=100,
	batch_size=128,
	validation_split=0.2)	
elif model_type=="LSTM":

	model = Sequential()
	model.add(Embedding(max_features, 128))
	model.add(LSTM(128))
	model.add(Dense(1, activation='softmax'))
	model.compile(optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy'])
	history = model.fit(x_train, y_train,
	epochs=10,
	batch_size=128,
	validation_split=0.2)

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
model.save_weights('pre_trained_glove_model.h5')

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
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


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