## Оглавление
- [Оглавление](#оглавление)
  - [1. Введение. Проблематика.](#1-введение-проблематика)
  - [2. Обучение модели на тестовых данных.](#2-обучение-модели-на-тестовых-данных)
  - [3. Работа с Flask (app.py)](#3-работа-с-flask-apppy)
  - [4. Разворачиваем наше приложение на Amvera](#4-разворачиваем-наше-приложение-на-amvera)
  - [5. Использование модели в Google Sheets](#5-использование-модели-в-google-sheets)


### 1. Введение. Проблематика.
  Для исключения влияния человека на категоризацию работ при бурении газовых скважин, была обучена модель, которая с точностью 97% классифицирует указанные работы. Данный подход позволит человеку перепроверить себя, а также на перспективу может увеличиться эффективность, а самое главное уменьшить количество ошибок. Каждая неправильно классифицирвоанная операция при буернии - это неверный расчет оплаты времени какому-либо из подрядчиков.

### 2. Обучение модели на тестовых данных.  
  Обучение модели производилось с помощью таких инструментов как **TfidfVectorizer** и **PassiveAggressiveClassifier**. Обучение проходило на [тестовой выборке](https://docs.google.com/spreadsheets/d/1yr_Bc2-Z9ABLrPWsjSAnQyMxMdXGWQo8lNoMDssxZH8/edit?usp=sharing) из датасэта (около 30 тысяч строк).
```
# Подготовка данных для обучения
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
```
Точность модели на тестовой выборке составила **97%.** 

Для дальнейшей работы,  допишем функцию, обрабатывающую входные тектстовые данные: 
```
# Функция для классификации работ
def classify_news(news_text):
    # Преобразование текста новости в числовые признаки
    tfidf_news = tfidf_vectorizer.transform([news_text])

    # Предсказание
    prediction = pac.predict(tfidf_news)

    # Возврат первого элемента из массива предсказаний
    return prediction[0]
```
### 3. Работа с Flask ([app.py](app.py))
Следующим этапом идет подготовка приложения во Flask. 
 - Импорт необходимых библиотек:
 - - Flask для создания веб-приложения
 - - joblib для загрузки сохраненных моделей
 - - os для работы с файловой системой 
 - - logging для ведения журнала
  
 - Настройка логирования для отслеживания работы приложения.
 - - Загрузка предварительно обученных моделей:
 - - TF-IDF векторизатор
 - - PassiveAggressiveClassifier
 - Определение маршрутов:
- - GET-запрос к корневому пути ('/') возвращает сообщение о работе сервера.
- - POST-запрос к '/classify2' для классификации текста.
- Функция классификации:
- - Принимает JSON-данные с полем 'text'.
- - Преобразует текст с помощью TF-IDF векторизатора.
- - Классифицирует текст с использованием PassiveAggressiveClassifier.
- - Возвращает результат классификации в формате JSON.
- Обработка ошибок:
- - Проверка корректности входных данных.
- - Логирование ошибок и предупреждений.
- - Запуск сервера:
- - Определение порта (по умолчанию 80).
- - Запуск Flask-приложения на указанном порту.

### 4. Разворачиваем наше приложение на Amvera
Для настройки сборки и запуска проекта можно создать файл amvera.yml в корне репозитория. Создавать amvera.yaml лучше всего в интерфейсе приложения в разделе «Конфигурация» так он автоматически добавится в корень проекта создав при этом новый коммит в гит репозиторий.
Основной репозитарий для файло можно создать в самом приложении Amvera. Я выбрал вариант с размещением на GitHub в связке через WebHook. 
Лог приложения при развертывании: 
```
[2024-08-04 14:12:35 +0000] [1] [INFO] Shutting down: Master
[2024-08-04 14:12:41 +0000] [1] [INFO] Starting gunicorn 22.0.0
[2024-08-04 14:12:41 +0000] [1] [INFO] Listening at: http://0.0.0.0:80 (1)
[2024-08-04 14:12:41 +0000] [1] [INFO] Using worker: sync
[2024-08-04 14:12:41 +0000] [7] [INFO] Booting worker with pid: 7
INFO:app:Модели успешно загружены
INFO:app:Получен POST-запрос к пути /classify2
INFO:app:Успешная классификация текста: 'Сборка КНБК в инт. 0-10м....
```

### 5. Использование модели в Google Sheets
Задумано, что проверку текстовых материалов с применением модели работник будет производить в таблице Google Sheets, которая находится в открытом доступе. 
Для реализации этой задумки был написан [скрипт](Model_goole_sheet_script.txt) в среде App Script. 
Структура [таблицы](https://docs.google.com/spreadsheets/d/1ODi3_E69mCoEHk_JWp92KP8jpbk4uxIx46iWXecOJPo/edit?usp=sharing) простая - в одном столбце текст для анализа, а во втором столбце результат функции  classify - ответ от модели. 
Скрипт будем запускать с кнопки в меню, для этого добоавим функцию OnOpen:
```
function onOpen(){
  var ui = SpreadsheetApp.getUi();
  ui.createMenu('Меню скриптов')
  .addItem('Классификация','classifyText')
  .addToUi();
```

