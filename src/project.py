#!/usr/bin/env python
# coding: utf-8

# In[49]:


# Import libraries
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

import fasttext
import pickle

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout


# # Read data

# In[2]:


# Read data
train = pd.read_csv("train.txt",delimiter=';',header=None,names=['sentence','label'])
test = pd.read_csv("test.txt",delimiter=';',header=None,names=['sentence','label'])


# In[3]:


df = pd.concat([train,test])
print(df.shape)
df.head()


# In[4]:


print(df['label'].value_counts())
print('\n')
print(df['label'].value_counts(normalize=True))


# In[5]:


print(train.shape)
print(train['label'].value_counts(normalize=True))


# In[6]:


print(test.shape)
print(test['label'].value_counts(normalize=True))


# # Preprocess text

# In[7]:


def clean_data(data):
    
    stop_words = set(stopwords.words('english'))
    text_clean = []
    new_data = pd.DataFrame({'sentence': data.sentence, 'label': data.label})
    
    new_data['sentence'] = data.apply(lambda r: ' '.join(w.lower() for w in r['sentence'].split() 
                                                         if (w.lower() not in stop_words) & 
                                                         (w.isalpha())),axis=1)
    new_data['sentence'] = new_data[new_data['sentence'] != '']
    new_data = new_data.dropna()

    return new_data


# In[8]:


def extract_features(train_set, test_set, ngram):

    tfidf=TfidfVectorizer(use_idf=True, max_df=0.95, ngram_range=ngram)
    tfidf.fit_transform(train_set['sentence'].values)

    train_tfidf=tfidf.transform(train_set['sentence'].values)
    test_tfidf=tfidf.transform(test_set['sentence'].values)

    return train_tfidf,test_tfidf,tfidf


# In[9]:


df_clean = clean_data(df)
df_clean.to_csv('data_clean.csv')

training_df, testing_df = train_test_split(df_clean[['sentence', 'label']].dropna(), 
                                               test_size = 0.2, random_state = 2020)

X_train, X_test, tfidf_vectorizer = extract_features(training_df, testing_df, (1,2))
y_train = training_df['label'].values
y_test = testing_df['label'].values

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)


# In[71]:


np.mean([len(i) for i in df_clean.sentence])


# In[73]:


y_train.shape


# # Fit models

# ## Logistic regression

# In[10]:


def predict_evaluate(model, test_tfidf, test_y):
    prediction = model.predict(test_tfidf)
    print(classification_report(test_y, prediction))
    return prediction


# In[11]:


lr1 = LogisticRegression(random_state=2020, C=15, penalty='l2', max_iter=1000, verbose=1)
lr1_classifier = OneVsRestClassifier(lr1)
model_lr1 = lr1_classifier.fit(X_train,y_train)

lr1_pred = predict_evaluate(model_lr1, X_test, y_test)
# print('Accuracy:', accuracy_score(Y_test, Y_pred))
# print('F1-score:', f1_score(Y_test, Y_pred, average='micro'))


# In[12]:


lr2 = LogisticRegression(random_state=2020, C=5, penalty='l2', max_iter=1000, verbose=1)
lr2_classifier = OneVsRestClassifier(lr2)
model_lr2 = lr2_classifier.fit(X_train,y_train)

lr2_pred = predict_evaluate(model_lr2, X_test, y_test)


# In[13]:


lr3 = LogisticRegression(random_state=2020, C=30, penalty='l2', max_iter=1000, verbose=1)
lr3_classifier = OneVsRestClassifier(lr3)
model_lr3 = lr3_classifier.fit(X_train,y_train)

lr3_pred = predict_evaluate(model_lr3, X_test, y_test)


# In[14]:


filepath = "logistic_best.pkl"
with open(filepath, 'wb') as file:
    pickle.dump(model_lr3, file)


# ## SVM

# In[15]:


svm1 = LinearSVC(random_state=2020, C=1, loss='squared_hinge', max_iter=1000)
model_svm1 = svm1.fit(X_train,y_train)

svm1_pred = predict_evaluate(model_svm1, X_test, y_test)


# In[16]:


svm2 = LinearSVC(random_state=2020, C=50, loss='squared_hinge', max_iter=1000)
model_svm2 = svm2.fit(X_train,y_train)

svm2_pred = predict_evaluate(model_svm2, X_test, y_test)


# In[17]:


svm3 = LinearSVC(random_state=2020, C=50, loss='hinge', max_iter=1000)
model_svm3 = svm3.fit(X_train,y_train)

svm3_pred = predict_evaluate(model_svm3, X_test, y_test)


# In[18]:


svm4 = LinearSVC(random_state=2020, C=1, loss='hinge', max_iter=1000)
model_svm4 = svm4.fit(X_train,y_train)

svm4_pred = predict_evaluate(model_svm4, X_test, y_test)


# In[19]:


filepath = "svm_best.pkl"
with open(filepath, 'wb') as file:
    pickle.dump(model_svm2, file)


# ## Fasttext

# In[22]:


# Format the data
train_fasttext = training_df.apply(lambda t: '__label__' + str(t['label']) + 
                                   ' ' + str(t['sentence']), axis=1)
test_fasttext = testing_df.apply(lambda t: '__label__' + str(t['label']) + 
                                 ' ' + str(t['sentence']), axis=1)
train_fasttext.to_csv('train_fasttext.txt',index=False, header=False)
test_fasttext.to_csv('test_fasttext.txt',index=False, header=False)


# In[23]:


model_ft1 = fasttext.train_supervised('train_fasttext.txt', loss='softmax',
                                      lr=0.1, ws=5, wordNgrams=2, epoch=100)
ft1_pred = model_ft1.test('test_fasttext.txt')

print("precision: ", ft1_pred[1])
print("recall: ", ft1_pred[2])
print("F-1 score: ", 2*ft1_pred[1]*ft1_pred[2]/(ft1_pred[1]+ft1_pred[2]))


# In[24]:


model_ft12 = fasttext.train_supervised('train_fasttext.txt', loss='softmax',
                                      lr=0.2, ws=10, wordNgrams=2, epoch=100)
ft2_pred = model_ft12.test('test_fasttext.txt')

print("precision: ", ft2_pred[1])
print("recall: ", ft2_pred[2])
print("F-1 score: ", 2*ft2_pred[1]*ft2_pred[2]/(ft2_pred[1]+ft2_pred[2]))


# In[25]:


model_ft13 = fasttext.train_supervised('train_fasttext.txt', loss='softmax',
                                      lr=0.2, ws=10, wordNgrams=2, epoch=300)
ft3_pred = model_ft13.test('test_fasttext.txt')

print("precision: ", ft3_pred[1])
print("recall: ", ft3_pred[2])
print("F-1 score: ", 2*ft3_pred[1]*ft3_pred[2]/(ft3_pred[1]+ft3_pred[2]))


# In[26]:


model_ft14 = fasttext.train_supervised('train_fasttext.txt', loss='softmax',
                                      lr=0.2, ws=15, wordNgrams=2, epoch=100)
ft4_pred = model_ft14.test('test_fasttext.txt')
ft4_pred
print("precision: ", ft4_pred[1])
print("recall: ", ft4_pred[2])
print("F-1 score: ", 2*ft4_pred[1]*ft4_pred[2]/(ft4_pred[1]+ft4_pred[2]))


# In[74]:


ft4_pred


# ## LSTM

# In[27]:


def load_data(file_path): 
    
    df = pd.read_csv(file_path)
    return df



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):    
    
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text) 
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwords from text
    return text

def process_texts(file_dir: str, is_train: bool) -> None:    
    all_texts = []    
    for file_name in os.listdir(file_dir):        
        if is_train and file_name.startswith('cv9'):            
            continue        
        if not is_train and not file_name.startswith('cv9'):            
            continue        
        file_path = os.path.join(file_dir, file_name)        
        cleaned_text = clean_text(load_text(file_path))
        all_texts.append(cleaned_text)    
    return all_texts


# In[28]:


def build_lstm_classifier(X_train, Y_train, X_test, Y_test,
                          MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):    
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(13, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
   


# In[29]:


file_path = 'data_clean.csv'
df = load_data(file_path) 

df['sentence'] = df['sentence'].apply(clean_text).str.replace('\d+', '')

print(df['sentence']) 


# In[32]:


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['sentence'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[35]:


X = tokenizer.texts_to_sequences(df['sentence'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# In[36]:


Y = pd.get_dummies(df['label']).values
print('Shape of label tensor:', Y.shape)


# In[37]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2020)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[50]:


def build_lstm_classifier(MAX_NB_WORDS, EMBEDDING_DIM, INPUT_LENGTH=X.shape[1], 
                          NEURONS=100, LR=0.01):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=INPUT_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(NEURONS, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LR), 
                  metrics=['accuracy'])
    print(model.summary())
    return model


# In[53]:


EPOCHS = 5
BACH_SIZE = 64

model1 = build_lstm_classifier(MAX_NB_WORDS, EMBEDDING_DIM, INPUT_LENGTH=X.shape[1], 
                               NEURONS=100, LR=0.001)

history1 = model1.fit(X_train, Y_train, 
                    epochs=EPOCHS, batch_size=BACH_SIZE,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[54]:


accr1 = model1.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr1[0],accr1[1]))


# In[55]:


EPOCHS = 5
BACH_SIZE = 64

model2 = build_lstm_classifier(MAX_NB_WORDS, EMBEDDING_DIM, INPUT_LENGTH=X.shape[1], 
                               NEURONS=200, LR=0.001)

history2 = model2.fit(X_train, Y_train, 
                    epochs=EPOCHS, batch_size=BACH_SIZE,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[56]:


accr2 = model2.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr2[0],accr2[1]))


# In[58]:


EPOCHS = 5
BACH_SIZE = 64

model3 = build_lstm_classifier(MAX_NB_WORDS, EMBEDDING_DIM, INPUT_LENGTH=X.shape[1], 
                               NEURONS=200, LR=0.0005)

history3 = model3.fit(X_train, Y_train, 
                    epochs=EPOCHS, batch_size=BACH_SIZE,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[59]:


accr3 = model3.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr3[0],accr3[1]))


# In[ ]:




