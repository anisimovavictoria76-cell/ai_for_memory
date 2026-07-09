from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import random
import database
import train_model
from recommender import WordRecommender
import os

app = Flask(__name__)

database.init_db()

try:
    with open('memory_model_enhanced.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Модель загружена")
except:
    model = None
    print("Модель не найдена. Сначала запустите train_model.py")

user_stats = {}


@app.route('/')
def index():
    """Главная страница с описанием и тренажером"""
    return render_template('index.html')


@app.route('/api/words', methods=['GET'])
def get_words():
    words = database.get_all_words()
    return jsonify(words)


@app.route('/api/words', methods=['POST'])
def add_word():
    data = request.json
    word = data.get('word', '').strip()
    translation = data.get('translation', '').strip()
    difficulty = int(data.get('difficulty', 1))

    print(f" Попытка добавить: word='{word}', translation='{translation}'")

    if not word or not translation:
        return jsonify({'error': 'Слово и перевод обязательны'}), 400

    success = database.add_word(word, translation, difficulty)
    print(f" Результат: success={success}")

    if success:
        try:
            print("Переобучение модели...")
            train_model.train_enhanced_model()
            with open('memory_model_enhanced.pkl', 'rb') as f:
                global model
                model = pickle.load(f)
            print(" Модель переобучена")
        except Exception as e:
            print(f" Ошибка переобучения: {e}")

    return jsonify({'success': success})


@app.route('/api/words', methods=['DELETE'])
def delete_word():
    data = request.json
    word = data.get('word', '').strip()

    if not word:
        return jsonify({'error': 'Слово обязательно'}), 400

    success = database.delete_word(word)
    return jsonify({'success': success})


@app.route('/api/start', methods=['POST'])
def start_session():
    user_id = request.json.get('user_id', 'default')
    words = database.get_all_words()

    if len(words) < 3:
        return jsonify({
            'error': 'Слишком мало слов. Добавьте минимум 3 слова для начала.'
        }), 400

    random.shuffle(words)

    user_stats[user_id] = {
        'words': words,
        'current_index': 0,
        'correct': 0,
        'wrong': 0,
        'difficult_words': []
    }

    return jsonify({
        'total_words': len(words),
        'first_word': words[0]
    })


@app.route('/api/next_word', methods=['POST'])
def next_word():
    user_id = request.json.get('user_id', 'default')
    last_result = request.json.get('last_result')
    last_word = request.json.get('last_word')

    if user_id not in user_stats:
        return jsonify({'error': 'Сессия не найдена'}), 400

    stats = user_stats[user_id]
    current_index = stats['current_index']
    words = stats['words']

    if last_result and last_word:
        if last_result == 'correct':
            stats['correct'] += 1
        else:
            stats['wrong'] += 1
            stats['difficult_words'].append(last_word)

    if current_index >= len(words):
        return jsonify({
            'finished': True,
            'correct': stats['correct'],
            'wrong': stats['wrong'],
            'difficult_words': stats['difficult_words']
        })

    word = words[current_index]
    stats['current_index'] += 1

    return jsonify({
        'word': word,
        'progress': current_index,
        'total': len(words)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    word = data.get('word', '')
    attempts = data.get('attempts', 1)
    difficulty = data.get('difficulty', 1)

    if model is None:
        return jsonify({'forget_probability': 0.5})

    forget_probability = train_model.predict_forget_probability(
        word, difficulty, attempts
    )

    return jsonify({
        'forget_probability': forget_probability,
        'difficulty': difficulty
    })


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    data = request.json
    session_id = data.get('session_id', 'default')
    limit = data.get('limit', 10)

    recommender = WordRecommender(session_id)
    recommendations = recommender.get_recommendation_for_session(limit=limit)

    return jsonify({
        'recommendations': recommendations,
        'count': len(recommendations)
    })


@app.route('/api/weekly_plan', methods=['POST'])
def get_weekly_plan():
    data = request.json
    session_id = data.get('session_id', 'default')

    recommender = WordRecommender(session_id)
    plan = recommender.get_weekly_plan()

    return jsonify(plan)


@app.route('/api/save_attempt', methods=['POST'])
def save_attempt():
    data = request.json
    word = data.get('word', '').strip()
    is_correct = data.get('is_correct', False)
    session_id = data.get('session_id', 'default')

    if not word:
        return jsonify({'error': 'Слово обязательно'}), 400

    database.save_attempt(word, is_correct, session_id)

    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)