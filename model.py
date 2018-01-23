import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#load ASCII data
input_file = "chinese_abc_2.txt"
raw_text = open(input_file).read()

#map unique chars to integers (also create reverse mapping)
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
print(chars)

#summarize loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

#prepare dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

#reshape X to be in the form [samples, time steps, features] for LSTM
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
#normalize
X = X/float(n_vocab)
#one hot encode the output variable
y = np_utils.to_categorical(dataY)

#define LSTM model
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

#define model checkpoint
#filepath = "weights-improvement-{epoch:02d}--{loss:.4f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only=True, mode = 'min')
#callbacks_list = [checkpoint]
#model.fit(X, y, epochs=40, batch_size=128, callbacks=callbacks_list)

#load network weights
filename = "weights-improvement-40--0.9797.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

#random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

#generate characters
for i in range(500):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]