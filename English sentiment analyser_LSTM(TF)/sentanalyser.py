from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras.losses import binary_crossentropy
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#import os

##################################################################

t=117 #max input shape
v=569373 #number of unique words
import pickle
with open('Module Objects/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

##################################################################

model=Sequential()
model.add(Embedding(v+1,64,input_length=t))
model.add(LSTM(128,dropout=0.2,return_sequences=True))
model.add(LSTM(128,dropout=0.2,return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dense(1,activation='sigmoid'))
model.compile(loss=binary_crossentropy,optimizer='adam')
model.load_weights("Module Objects/sentiment_weights.03.hdf5")

##################################################################

def stranalyse(st):                               
    st=[st]
    st=tokenizer.texts_to_sequences(st)
    st=pad_sequences(st,maxlen=t)
    pr=model.predict(st,verbose=0)
    if pr[0][0]>0.6: return 'happy'
    elif pr[0][0]>0.4: return 'neutral'
    else: return 'sadness'

##################################################################

def seranalyse(df):
    sents=[]
    df=tokenizer.texts_to_sequences(df)
    df=pad_sequences(df,maxlen=t)
    pr=model.predict(df,verbose=0)
    for doc in pr:
        if doc[0]>0.6: sents.append('happy')
        elif doc[0]>0.4: sents.append('neutral')
        else: sents.append('sadness')
    return sents