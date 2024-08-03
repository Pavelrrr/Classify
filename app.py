from flask import Flask, request, jsonify
import joblib
import os
import logging
from werkzeug.exceptions import BadRequest
###
app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    pac_classifier = joblib.load('pac_classifier.joblib')
    logger.info("Модели успешно загружены")
except Exception as e:
    logger.error(f"Ошибка при загрузке моделей: {str(e)}")
    raise

@app.route('/', methods=['GET'])
def home():
    logger.info("Получен GET-запрос к корневому пути")
    return jsonify({"message": "Сервер работает. Используйте путь /classify для классификации текста."})

@app.route('/classify2', methods=['POST'])
def classify():
    logger.info("Получен POST-запрос к пути /classify2")
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            raise BadRequest("Отсутствует поле 'text' в JSON-данных")
        
        text = data['text']
        tfidf_text = tfidf_vectorizer.transform([text])
        prediction = pac_classifier.predict(tfidf_text)
        
        logger.info(f"Успешная классификация текста: '{text[:50]}...'")
        return jsonify({'classification': prediction[0]})
    except BadRequest as e:
        logger.warning(f"Некорректный запрос: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Ошибка при классификации: {str(e)}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    logger.info(f"Запуск сервера на порту {port}")
    app.run(host='0.0.0.0', port=port)
