#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[83]:


df = pd.read_csv('./ner_dataset.csv', encoding='latin1')
df = df.fillna(method='ffill')
words = list(set(df['Word'].values))
words.append('endpad')
tags = list(set(df['Tag'].values))
len(words), len(tags)


# In[84]:


agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
grouped = df.groupby("Sentence #").apply(agg_func)
sentences = [s for s in grouped]
sentences[0]


# In[85]:


max_len = max([len(sentence) for sentence in sentences])
max_len


# In[86]:


wordtoidx = {word:idx for idx,word in enumerate(words)}
tagtoidx = {tag:idx for idx,tag in enumerate(tags)}


# In[48]:


X = [[wordtoidx[word[0]] for word in sentence] for sentence in sentences]
Y = [[tagtoidx[word[2]] for word in sentence] for sentence in sentences]


# In[26]:


X = pad_sequences(sequences=X, maxlen=max_len, padding='post', value=len(words)-1)
Y = pad_sequences(sequences=Y, maxlen=max_len, padding='post', value=tagtoidx["O"])


# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
len(X_train), len(Y_train)
X_test[2]


# In[29]:


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional


# In[31]:


# creating the model
input_data = Input(shape=(max_len,))
model = Embedding(input_dim=len(words), output_dim=50, input_length=max_len)(input_data)
model = SpatialDropout1D(rate=0.1)(model)
model = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(units=len(tags), activation='softmax'))(model)
model = Model(input_data, out)
model.summary()


# In[32]:


model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')


# In[33]:


model.fit(
    x=X_train,
    y=Y_train,
    batch_size=32,
    validation_data=(X_test,Y_test),
    epochs=3,
    verbose=1
)


# In[34]:


model.save('./')


# In[77]:


model.evaluate(X_test,Y_test)


# In[124]:


# Preprocessing User Input
def word_to_idx(words_predict):
    words_idx = np.full((max_len,), len(words)-1)
    i = 0
    for w in words_predict:
        words_idx[i] = wordtoidx[w]
        i += 1
    return words_idx

input_sentence = "I want to fly in an Airbus. I am planning a trip to London"
words_predict = list(set(word_tokenize(input_sentence)))
x_predict = word_to_idx(words_predict)
p = model.predict(np.array(x_predict))
p = np.argmax(p, axis=-1)
for i in range(len(p)):
    print("{} - {}".format(words[x_predict[i]], tags[p[i][0]]))


# In[ ]:




