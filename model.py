import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional,Conv1D,MaxPooling1D,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
#from tensorflow.keras.utils import to_categorical

train_data=pd.read_csv('train_data.csv')
test_data=pd.read_csv('test_data.csv')

train_data=train_data.dropna()
test_data=test_data.dropna()

train_data.Sentiment=train_data.Sentiment.apply(lambda x:1 if x=='Positive' else (0 if x=='Neutral' else -1))
test_data.Sentiment=test_data.Sentiment.apply(lambda x:1 if x=='Positive' else (0 if x=='Neutral' else -1))

train_size=int(len(train_data) * 0.75)

x_train=train_data.train_cleaned
y_train=train_data.Sentiment

x_test=test_data.test_cleaned
y_test=test_data.Sentiment


max_len = np.max(x_train.apply(lambda x :len(x)))
print("Max sequence length:", max_len)

#converting text to numeric using tokenizer with padding sequences to max length
tokenizer=Tokenizer(num_words=4500)
tokenizer.fit_on_texts(x_train)


train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post',truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post',truncating='post')


#one hot encoding the target variables
#y_train = to_categorical(y_train, 3)
#y_test = to_categorical(y_test, 3)

#custom function containing bidirectional LSTM with dropout
def create_model(embedding_dim=30,filters=32,kernel_size=3,lstm_units=64,dropout=0.2,num_layers=1,
                 dense_units=32,init_mode='uniform'):
    model = Sequential()
    model.add(Embedding(input_dim=4500,output_dim=embedding_dim,input_length=max_len)) #The embedding layer
    model.add(Conv1D(filters=filters,
                     kernel_size=kernel_size, padding='same', activation='relu',
                     kernel_initializer=init_mode))# 1D Convolutional layer
    model.add(MaxPooling1D())#max pooling layer
    model.add(Bidirectional(LSTM(units=lstm_units,dropout=dropout,kernel_initializer=init_mode))) #Our LSTM layer
    for i in range(num_layers):
         model.add(Dense(units=dense_units,activation='relu',kernel_initializer=init_mode))#Dense layer with 32 units with relu activation
         model.add(Dropout(dropout))
    model.add(Dense(3,activation='softmax',kernel_initializer=init_mode))#Dense layer for classifying 3 classes by use of softmax activation
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

keras_estimator = KerasClassifier(build_fn=create_model,epochs=3,batch_size=64,verbose=1)

cv=StratifiedKFold(n_splits=2, shuffle=True)

params_grid={
        'embedding_dim':[25,30,35,40,45],
         'filters':[16,32,64],
         'kernel_size':[3,5,7],
         'lstm_units':[32,64,128],
         'dropout':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
         'num_layers':[1,2],
         'dense_units':[32,64,128],
         'init_mode' : ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
                        'glorot_uniform', 'he_normal', 'he_uniform']
         }

rs = RandomizedSearchCV(estimator=keras_estimator,   
                    verbose=1,
                    return_train_score=True,
                    cv=cv,  #
                    param_distributions=params_grid)

rs_result = rs.fit(train_padded[:train_size],y_train[:train_size]) 

# summarize results
print("Best: %f using %s" % (rs_result.best_score_, rs_result.best_params_))
means = rs_result.rs_results_['mean_test_score']
stds = rs_result.rs_results_['std_test_score']
params = rs_result.rs_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
best_model=rs_result.best_estimator_
best_model.fit(train_padded,y_train)


pred = best_model.predict(test_padded)
cm = confusion_matrix(y_test,pred)
plt.figure(figsize=(10,5))
sns.heatmap(cm,annot=True)

print(classification_report(y_test,pred,target_names=['Negative','Neutral','Positive']))








































