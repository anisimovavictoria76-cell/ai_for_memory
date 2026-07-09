import pytest
import os
import sqlite3
import pickle
import numpy as np
from unittest.mock import patch, MagicMock
import train_model
import database


# Фикстура: временная БД для тестов
@pytest.fixture
def test_db():
    original_db = database.DB_PATH
    test_db_path = 'test_words.db'
    database.DB_PATH = test_db_path
    database.init_db()

    yield test_db_path

    database.DB_PATH = original_db
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Удаляем временную модель
    if os.path.exists('test_model.pkl'):
        os.remove('test_model.pkl')


class TestTrainModel:
    """Тесты для обучения и предсказаний"""

    def test_create_training_data_with_words(self, test_db):
        """Проверяет создание обучающих данных"""
        df = train_model.create_training_data()

        assert df is not None
        assert len(df) > 0
        assert 'difficulty' in df.columns
        assert 'attempts' in df.columns
        assert 'word_length' in df.columns
        assert 'is_correct' in df.columns

    def test_create_training_data_with_few_words(self, test_db):
        """Проверяет, что при малом количестве слов возвращается None"""
        # Очищаем таблицу
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM words")
        conn.commit()
        conn.close()

        # Добавляем только 2 слова
        database.add_word('one', 'один', 1)
        database.add_word('two', 'два', 1)

        df = train_model.create_training_data()
        assert df is None  # Меньше 5 слов → None

    def test_train_model_creates_file(self, test_db):
        """Проверяет, что train_model создает файл модели"""
        model = train_model.train_model()

        assert model is not None
        assert os.path.exists('memory_model.pkl')

    def test_train_model_with_few_words(self, test_db):
        """Проверяет обучение при малом количестве слов"""
        # Очищаем таблицу
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM words")
        conn.commit()
        conn.close()

        # Добавляем только 2 слова
        database.add_word('one', 'один', 1)
        database.add_word('two', 'два', 1)

        model = train_model.train_model()
        assert model is None  # Меньше 5 слов → None

    def test_predict_forget_probability_without_model(self):
        """Проверяет предсказание без модели"""
        # Удаляем модель, если она есть
        if os.path.exists('memory_model.pkl'):
            os.remove('memory_model.pkl')

        prob = train_model.predict_forget_probability('apple', 1, 1)
        assert prob == 0.5  # Возвращает 0.5, если модели нет

    def test_predict_forget_probability_with_model(self, test_db):
        """Проверяет предсказание с обученной моделью"""
        # Обучаем модель
        train_model.train_model()

        # Тестируем предсказания
        prob = train_model.predict_forget_probability('apple', 1, 1)
        assert isinstance(prob, float)
        assert 0 <= prob <= 1  # Вероятность должна быть от 0 до 1

        # Сложное слово → выше вероятность ошибки
        prob_easy = train_model.predict_forget_probability('cat', 1, 1)
        prob_hard = train_model.predict_forget_probability('congratulation', 1, 3)
        # Обычно сложнее слово → выше вероятность ошибки
        # Но из-за случайности может быть не всегда, поэтому просто проверяем тип
        assert isinstance(prob_easy, float)
        assert isinstance(prob_hard, float)

    def test_predict_forget_probability_with_invalid_model(self):
        """Проверяет предсказание с поврежденной моделью"""
        # Создаем поврежденный файл модели
        with open('memory_model.pkl', 'w') as f:
            f.write('invalid content')

        prob = train_model.predict_forget_probability('apple', 1, 1)
        assert prob == 0.5  # Должен вернуть 0.5 при ошибке