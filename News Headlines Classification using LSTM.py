# استيراد المكتبات الأساسية
import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# تحميل قائمة الكلمات الشائعة (Stopwords) من مكتبة NLTK
nltk.download('stopwords')

# ربط Google Colab مع Google Drive للوصول للملفات
from google.colab import drive
drive.mount('/content/drive')

# قراءة ملف البيانات (CSV) من Drive
df = pd.read_csv('/content/drive/MyDrive/train.csv')

# إزالة أي صفوف تحتوي على قيم فارغة
df.dropna(inplace=True)
# إعادة ترقيم الصفوف بعد حذف الصفوف الفارغة
df.reset_index(drop=True, inplace=True)

# عرض أول 10 صفوف للتأكد من البيانات
df.head(10)

# تحديد العمود الذي يحتوي على النصوص (X) والعمود الذي يحتوي على التصنيفات (y)
X = df['title']   # نصوص العناوين
y = df['label']   # التصنيفات (Binary: 0 أو 1)

# تهيئة PorterStemmer لمعالجة الكلمات (اختصارها لجذر الكلمة)
ps = PorterStemmer()
corpus = []

# تنظيف ومعالجة النصوص
for i in range(len(X)):
    text = X[i]
    # إزالة أي رموز غير حروف
    text = re.sub('[^a-zA-Z]', ' ', text)
    # تحويل النصوص إلى أحرف صغيرة
    text = text.lower()
    # تقسيم النص إلى كلمات
    text = text.split()
    # إزالة الكلمات الشائعة وتطبيق stemming
    text = [ps.stem(t) for t in text if t not in stopwords.words('english')]
    # إعادة دمج الكلمات في نص واحد
    corpus.append(' '.join(text))

# إعداد المعلمات للشبكة العصبية
vocab_size = 5000  # حجم المفردات: عدد الكلمات الفريدة التي سنرمزها
sent_len = 20      # طول كل جملة بعد التوسيع أو القص (Padding/Truncating)

# تحويل النصوص إلى أرقام باستخدام One-hot encoding
one_hot_encoded = [one_hot(x, vocab_size) for x in corpus]

# توحيد طول الجمل باستخدام Padding (لتصبح كل الجمل بنفس الطول)
one_hot_encoded = pad_sequences(one_hot_encoded, maxlen=sent_len)

# تحويل البيانات إلى مصفوفات numpy
X = np.array(one_hot_encoded)
y = np.array(y)

# تقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# عدد الخصائص (Features) في طبقة Embedding
no_of_output_features = 40

# بناء نموذج LSTM باستخدام Keras
model = Sequential()
# طبقة Embedding: تحويل كل كلمة إلى تمثيل رقمي كثيف (Vector)
model.add(Embedding(vocab_size, no_of_output_features, input_length=sent_len))
# طبقة Dropout لتقليل الإفراط في التعلّم (Overfitting)
model.add(Dropout(0.5))
# طبقة LSTM لمعالجة تسلسل النصوص
model.add(LSTM(100))
# Dropout إضافي بعد LSTM
model.add(Dropout(0.5))
# طبقة Dense نهائية لإنتاج تصنيف واحد (Binary classification)
model.add(Dense(1, activation='sigmoid'))  # sigmoid لتصنيف ثنائي

# تجميع النموذج مع تحديد optimizer و loss function ومقاييس الأداء
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# عرض ملخص النموذج
model.summary()

# تدريب النموذج على بيانات التدريب والتحقق من الأداء على بيانات الاختبار
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=64,
    epochs=40
)

# التنبؤ بالتصنيفات لمجموعة الاختبار
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # >0.5 لإعطاء 0 أو 1

# حساب وعرض مصفوفة الالتباس (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# حساب دقة النموذج (Accuracy)
accuracy = np.sum(y_pred.flatten() == y_test) / len(y_test)
print("Accuracy:", accuracy)
