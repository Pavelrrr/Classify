# Classify
- [Classify](#classify)


Проект создавался в обучающе-развивающих целях. 
Основная задумка: 
1. Проблематика
   Для исключения влияния человека на категоризацию работ при бурении газовых скважин, была обучена модель, которая с точностью 97% классифицирует указанные работы. Данный подход позволит человеку перепроверить себя, а также на перспективу может увеличиться эффективность, а самое главное уменьшить количество ошибок. Каждая неправильно классифицирвоанная операция при буернии - это неверный расчет оплаты времени какому-либо из подрядчиков. 
##2. Обучение модели на тестовых данных 
  Обучение модели производилось с помощью таких инструментов как **TfidfVectorizer** и **PassiveAggressiveClassifier**. Обучение проходило на тестовой выборке из датасэта (около 30 тысяч строк).

`  # Подготовка данных для обучения
X = combined_text
y = detail_df['Операция']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Импорт библиотеки для работы со стоп-словами
import nltk
from nltk.corpus import stopwords

# Загрузка стоп-слов для русского языка
nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')

# Вывод списка стоп-слов
print(russian_stopwords)

# Создание векторизатора TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words=russian_stopwords, max_df=0.7)

# Преобразование текстовых данных в числовые признаки
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Получение и вывод списка признаков
feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names)
print(tfidf_train)

# Инициализация и обучение PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Предсказание на тестовой выборке
y_pred = pac.predict(tfidf_test)

# Расчет и вывод точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность: {accuracy:.4f}') `

