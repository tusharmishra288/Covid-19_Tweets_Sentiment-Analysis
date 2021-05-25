#importing necessary libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix,classification_report


#loading train and test sets
train_data=pd.read_csv('train_data.csv')
test_data=pd.read_csv('test_data.csv')

#dropping the missing values from both train and test sets
train_data=train_data.dropna()
test_data=test_data.dropna()

#hard coding the labels into numbers
train_data.Sentiment=train_data.Sentiment.apply(lambda x:2 if x=='Positive' else (1 if x=='Neutral' else 0))
test_data.Sentiment=test_data.Sentiment.apply(lambda x:2 if x=='Positive' else (1 if x=='Neutral' else 0))

x_train=train_data.train_cleaned
y_train=train_data.Sentiment

x_test=test_data.test_cleaned
y_test=test_data.Sentiment

#calculating maximum length of sequences
max_len = np.max(x_train.apply(lambda x :len(x)))
print("Max sequence length:", max_len)

#converting text to vectors using tokenizer with padding sequences to maximum length
tokenizer=Tokenizer()

#fitting on training set
tokenizer.fit_on_texts(x_train)

#saving the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#number of unique words in train dataset
vocab_length = len(tokenizer.word_index) + 1

train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

#loading the glove pre-trained embeddings 
embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

#mapping pre-trained embedding according to our corpus
embedding_matrix = np.zeros((vocab_length, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
           
#model containing embedding,bidirectional LSTM,dense and dropout layers with softmax activation for multiclass classification
model = Sequential()
model.add(Embedding(input_dim=vocab_length,output_dim=100,weights=[embedding_matrix],
                    input_length=max_len)) 
model.add(Bidirectional(LSTM(units=64,dropout=0.7)))
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(train_padded,y_train,epochs=10,batch_size=64,validation_split=0.2)

#evaluation of best model on test set
model.evaluate(test_padded,y_test)

#visualizing predictions on test through classification report and confusion matrix
pred = np.argmax(model.predict(test_padded),axis=-1)
cm = confusion_matrix(np.array(y_test),pred)
plt.figure(figsize=(10,5))
sns.heatmap(cm,annot=True)
print(classification_report(np.array(y_test),pred,target_names=['Negative','Neutral','Positive']))

#saving the model as hdf5 file
model.save('saved_model')


























