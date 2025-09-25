import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.metrics import confusion_matrix,accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
nltk.download('stopwords')

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/train.csv')
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.head(10)

X = df['title']
y = df['label']

ps = PorterStemmer()
corpus = []
for i in range(len(X)):
    text = X[i]
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(t) for t in text if t not in stopwords.words('english')]
    corpus.append(' '.join(text))


vocab_size = 5000
sent_len = 20
one_hot_encoded = [one_hot(x,vocab_size) for x in corpus]
one_hot_encoded = pad_sequences(one_hot_encoded,maxlen=sent_len)
one_hot_encoded[0]


X = np.array(one_hot_encoded)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


no_of_output_features = 40

model = Sequential()

model.add(Embedding(vocab_size,no_of_output_features,input_length=sent_len))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()



model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=64,epochs=40)




!pip install scikit-learn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()
y_test_bin = label_binarizer.fit_transform(y_test)

pred = np.array([0, 1, 2, 3, 4])

cm = confusion_matrix(y_test_bin.argmax(axis=1), pred.argmax(axis=1))
print(cm)
