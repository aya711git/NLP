## 📰 تصنيف عناوين الأخبار باستخدام LSTM

🌟 نظرة عامة

مرحبًا بك في هذا المشروع! 😄
هنا نقوم ببناء نموذج تعلم عميق لتصنيف عناوين الأخبار إلى فئتين (0 أو 1) باستخدام شبكات الذاكرة طويلة المدى (LSTM).

الأهداف الرئيسية:

✨ معالجة النصوص: تنظيفها، إزالة الكلمات الشائعة، واختصار الكلمات لجذورها.

🧮 تمثيل النصوص رقميًا: باستخدام One-hot Encoding.

🧠 تعلم النموذج: تدريب LSTM لتصنيف النصوص القصيرة.

📊 تقييم الأداء: استخدام مصفوفة الالتباس والدقة.

## 📂 هيكل المستودع
```text

├── README.md                  # هذا الملف
├── train.csv                  # نموذج البيانات للتدريب
├── News Headlines Classification using LSTM.py  # الكود الرئيسي/دفتر python
└── LICENSE                    # ملف الترخيص
```

## 🗂️ البيانات

يجب أن يحتوي train.csv على عمودين رئيسيين:

title → النص أو العنوان المراد تصنيفه

label → التصنيف الثنائي (0 أو 1)

💡 ملاحظة: تأكد من عدم وجود قيم مفقودة، أو سيتم حذفها تلقائيًا أثناء المعالجة.

## 🚀 طريقة التشغيل

ربط Google Drive (إذا كنت تستخدم Colab):

from google.colab import drive
drive.mount('/content/drive')


تثبيت المكتبات اللازمة:

pip install nltk tensorflow pandas scikit-learn


تشغيل الكود لتقوم بالخطوات التالية:

🧹 تنظيف النصوص (إزالة stopwords وتطبيق Stemming)

🔢 تحويل النصوص إلى أرقام

🤖 بناء وتدريب نموذج LSTM

📈 تقييم أداء النموذج

مثال للتنبؤ بالتصنيفات:

y_pred = (model.predict(X_test) > 0.5).astype("int32")

## 🏗️ بنية النموذج

Embedding Layer: تحويل الكلمات إلى تمثيلات رقمية كثيفة.

LSTM Layer: معالجة تسلسل الكلمات والتعرف على الأنماط الزمنية.

Dropout Layers: لتجنب الإفراط في التعلّم (Overfitting).

Dense Layer: إخراج التصنيف النهائي (0 أو 1).

## 📊 التقييم

مصفوفة الالتباس (Confusion Matrix)

دقة النموذج (Accuracy)

مثال:

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

## 🎯 الترخيص

هذا المشروع مرخص تحت MIT License.
يمكنك استخدامه وتعديله وتوزيعه بحرية. ✅

## 💡 نصائح لتحسين الأداء

🔹 زيادة حجم المفردات (vocab_size)

🔹 تعديل عدد وحدات LSTM

🔹 تغيير طول التسلسل (sent_len)

🔹 تجربة تقنيات معالجة نصوص مختلفة

## 🏷️الكلمات المفتاحية:
python
nlp
natural-language-processing
text-classification
news-classification
lstm
deep-learning
machine-learning
tensorflow
keras
beginner-project


## 📌 ملاحظة ممتعة:
يمكنك تجربة هذا النموذج على أي مجموعة بيانات نصية قصيرة أخرى، كالعناوين أو التغريدات أو التعليقات، وسيعمل بنفس الطريقة! 🚀
