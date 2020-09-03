import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
df = pd.read_csv(r"C:\Users\rakes\Downloads\train\train.csv")
pd.set_option("display.max_columns", 5)

df = df.dropna()

X = df.drop("label", axis=1)
y = df['label']

### Vocabulary size
voc_size = 5000

message = X.copy()

message.reset_index(inplace=True)

from nltk.stem.porter import PorterStemmer

#Data preprocessing
ps = PorterStemmer()
corpus = []
for i in range(0,len(message)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', message['title'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words("english")]
    review = " ".join(review)
    corpus.append(review)

onehot_rep = [one_hot(words,voc_size) for words in corpus]

sent_length = 20
embedded_docs = pad_sequences(onehot_rep,padding='pre', maxlen=sent_length)


#creating model

embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

y_pred = model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))