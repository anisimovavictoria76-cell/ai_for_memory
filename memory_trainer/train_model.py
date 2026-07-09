import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import database
import re

database.init_db()

def extract_features(word):
    """Извлекает дополнительные признаки из слова"""
    return {
        'length': len(word),
        'unique_letters': len(set(word.lower())),
        'vowels': len(re.findall(r'[aeiouyаеёиоуыэюя]', word.lower())),
        'consonants': len(re.findall(r'[bcdfghjklmnpqrstvwxyzбвгджзйклмнпрстфхцчшщ]', word.lower())),
        'syllables': len(re.findall(r'[aeiouyаеёиоуыэюя]{1}', word.lower()))
    }


def create_enhanced_training_data():
    """Создает улучшенные данные для обучения нейросети"""

    words = database.get_all_words()
    if len(words) < 5:
        print(" Слишком мало слов. Добавьте минимум 5 слов.")
        return None

    data = []

    for word_data in words:
        word = word_data['word']
        difficulty = word_data['difficulty']

        # Извлекание дополнительных признаков
        features = extract_features(word)

        for attempt in range(50):
            # Повышенная сложность для слов с большим количеством уникальных букв
            difficulty_factor = difficulty * (1 + features['unique_letters'] / 20)

            # Вероятности на ошибки
            error_prob = 0.15 + (difficulty_factor - 1) * 0.15

            # Корректировка на длину и уникальность
            error_prob += (features['length'] / 30) * 0.1
            error_prob += (features['unique_letters'] / 15) * 0.1

            # Ограничиваем вероятность
            error_prob = min(error_prob, 0.9)

            is_correct = 1 if np.random.random() > error_prob else 0

            # Симулирование количества попыток
            attempts_count = np.random.randint(1, 8 + difficulty * 3)

            # запись
            data.append({
                'word': word,
                'difficulty': difficulty,
                'attempts': attempts_count,
                'word_length': features['length'],
                'unique_letters': features['unique_letters'],
                'vowels': features['vowels'],
                'consonants': features['consonants'],
                'syllables': features['syllables'],
                'is_correct': is_correct
            })

    df = pd.DataFrame(data)
    print(f"Создано {len(df)} примеров для обучения")
    return df


def train_enhanced_model():
    """Обучает улучшенную нейросеть"""

    print("Начинаем обучение улучшенной нейросети...")

    df = create_enhanced_training_data()
    if df is None or len(df) < 20:
        print("Недостаточно данных для обучения.")
        return None

    # Признаки для обучения
    feature_columns = [
        'difficulty', 'attempts', 'word_length',
        'unique_letters', 'vowels', 'consonants', 'syllables'
    ]

    X = df[feature_columns].values
    y = df['is_correct'].values

    # Масштаб. признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Сохранение scaler для использования в предсказаниях
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Деление на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Улучшенная архитектура нейросети
    model = MLPClassifier(
        hidden_layer_sizes=(24, 48, 24),  # 3 скрытых слоя
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 регуляризация (борьба с переобучением)
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=2000,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1
    )

    print(" Обучение модели...")
    model.fit(X_train, y_train)

    # Оценивание точности
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f" Точность на обучении: {train_accuracy * 100:.2f}%")
    print(f" Точность на тесте: {test_accuracy * 100:.2f}%")

    # Проверка на переобучение
    if train_accuracy - test_accuracy > 0.15:
        print(" Есть признаки переобучения. Возможно, нужно больше данных.")

    # Сохранение модели
    with open('memory_model_enhanced.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(" Улучшенная модель сохранена в memory_model_enhanced.pkl")

    return model


def predict_with_enhanced_model(word, difficulty, attempts):
    """Предсказывает вероятность ошибки с улучшенной моделью"""

    try:
        with open('memory_model_enhanced.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # признаки для слова
        features = extract_features(word)

        # вектор признаков
        feature_vector = np.array([[
            difficulty,
            attempts,
            features['length'],
            features['unique_letters'],
            features['vowels'],
            features['consonants'],
            features['syllables']
        ]])

        # Масштаб
        feature_vector = scaler.transform(feature_vector)

        # Предсказание
        prob = model.predict_proba(feature_vector)[0]
        return float(prob[0])  # вероятность ошибки

    except FileNotFoundError:
        return 0.5  # Если модели нет


if __name__ == '__main__':
    train_enhanced_model()