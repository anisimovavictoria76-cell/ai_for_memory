import pytest
import os
import sqlite3
import database


# Фикстура: временная база данных для тестов
@pytest.fixture
def test_db():
    """Создает временную БД для тестов и удаляет после"""
    # Сохраняем путь к оригинальной БД
    original_db = database.DB_PATH

    # Создаем тестовую БД
    test_db_path = 'test_words.db'
    database.DB_PATH = test_db_path

    # Инициализируем тестовую БД
    database.init_db()

    yield test_db_path

    # Очистка после тестов
    database.DB_PATH = original_db
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


class TestDatabase:
    """Тесты для базы данных"""

    def test_init_db_creates_tables(self, test_db):
        """Проверяет, что init_db создает таблицы"""
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()

        # Проверяем, что таблица words существует
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='words'"
        )
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == 'words'

        conn.close()

    def test_init_db_adds_default_words(self, test_db):
        """Проверяет, что добавляются слова по умолчанию"""
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM words")
        count = cursor.fetchone()[0]

        # Должно быть 16 слов по умолчанию
        assert count == 16

        conn.close()

    def test_get_all_words_returns_list(self, test_db):
        """Проверяет, что get_all_words возвращает список слов"""
        words = database.get_all_words()

        assert isinstance(words, list)
        assert len(words) == 16  # 16 слов по умолчанию
        assert 'word' in words[0]
        assert 'translation' in words[0]
        assert 'difficulty' in words[0]

    def test_add_word_success(self, test_db):
        """Проверяет добавление нового слова"""
        success = database.add_word('test_word', 'тестовое слово', 2)
        assert success is True

        words = database.get_all_words()
        assert len(words) == 17  # 16 + 1 новое

        # Проверяем, что слово добавилось правильно
        added = next(w for w in words if w['word'] == 'test_word')
        assert added['translation'] == 'тестовое слово'
        assert added['difficulty'] == 2

    def test_add_word_duplicate_fails(self, test_db):
        """Проверяет, что дубликат не добавляется"""
        # Добавляем слово
        database.add_word('unique_word', 'уникальное слово', 1)

        # Пытаемся добавить такое же
        success = database.add_word('unique_word', 'другой перевод', 2)
        assert success is False

        words = database.get_all_words()
        unique_words = [w for w in words if w['word'] == 'unique_word']
        assert len(unique_words) == 1  # Должен быть только один

    def test_add_word_empty_fails(self, test_db):
        """Проверяет, что пустые слова не добавляются"""
        success = database.add_word('', 'пустое слово', 1)
        assert success is False

        success = database.add_word('непустое', '', 1)
        assert success is False

    def test_delete_word_success(self, test_db):
        """Проверяет удаление слова"""
        # Добавляем слово для удаления
        database.add_word('delete_me', 'удали меня', 1)

        success = database.delete_word('delete_me')
        assert success is True

        words = database.get_all_words()
        assert not any(w['word'] == 'delete_me' for w in words)

    def test_delete_word_not_found(self, test_db):
        """Проверяет удаление несуществующего слова"""
        success = database.delete_word('non_existent_word')
        assert success is False

    def test_get_words_count(self, test_db):
        """Проверяет подсчет количества слов"""
        count = database.get_words_count()
        assert count == 16  #по умолчанию

        # Добавляем слово и проверяем
        database.add_word('count_test', 'тест счета', 1)
        count = database.get_words_count()
        assert count == 17