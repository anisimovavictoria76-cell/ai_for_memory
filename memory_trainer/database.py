import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = 'words.db'


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE,
            translation TEXT,
            difficulty INTEGER DEFAULT 1
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT,
            is_correct INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT
        )
    ''')

    cursor.execute('SELECT COUNT(*) FROM words')
    if cursor.fetchone()[0] == 0:
        words = [
            ('apple', 'яблоко', 1),
            ('dog', 'собака', 1),
            ('cat', 'кошка', 1),
            ('house', 'дом', 1),
            ('car', 'машина', 1),
            ('book', 'книга', 1),
            ('tree', 'дерево', 1),
            ('sun', 'солнце', 1),
            ('moon', 'луна', 1),
            ('star', 'звезда', 1),
            ('happiness', 'счастье', 2),
            ('beautiful', 'красивый', 2),
            ('computer', 'компьютер', 2),
            ('telephone', 'телефон', 2),
            ('university', 'университет', 3),
            ('congratulation', 'поздравление', 3),
        ]
        cursor.executemany('INSERT INTO words (word, translation, difficulty) VALUES (?, ?, ?)', words)

    conn.commit()
    conn.close()


def get_all_words():
    """Возвращает все слова из базы"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT word, translation, difficulty FROM words', conn)
    conn.close()
    return df.to_dict('records')


def add_word(word, translation, difficulty=1):
    """Добавляет новое слово в базу"""
    if not word or not translation:
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            'INSERT INTO words (word, translation, difficulty) VALUES (?, ?, ?)',
            (word.strip().lower(), translation.strip().lower(), difficulty)
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False

    conn.close()
    return success


def delete_word(word):
    """Удаляет слово из базы"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM words WHERE word = ?', (word.strip().lower(),))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def get_words_count():
    """Возвращает количество слов в базе"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM words')
    count = cursor.fetchone()[0]
    conn.close()
    return count


def save_attempt(word, is_correct, session_id='default'):
    """Сохраняет попытку пользователя"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO attempts (word, is_correct, session_id) VALUES (?, ?, ?)',
        (word.lower().strip(), 1 if is_correct else 0, session_id)
    )
    conn.commit()
    conn.close()


def get_word_stats(word):
    """Возвращает статистику по слову"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(is_correct) as correct,
            SUM(1 - is_correct) as wrong
        FROM attempts 
        WHERE word = ?
    ''', (word.lower().strip(),))
    result = cursor.fetchone()
    conn.close()

    if result and result[0] > 0:
        return {
            'total': result[0],
            'correct': result[1] or 0,
            'wrong': result[2] or 0,
            'accuracy': round((result[1] or 0) / result[0] * 100, 1)
        }
    return None


def get_all_words_stats():
    """Возвращает статистику по всем словам"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            word,
            COUNT(*) as total,
            SUM(is_correct) as correct,
            SUM(1 - is_correct) as wrong
        FROM attempts 
        GROUP BY word
    ''')
    results = cursor.fetchall()
    conn.close()

    stats = {}
    for row in results:
        stats[row[0]] = {
            'total': row[1],
            'correct': row[2] or 0,
            'wrong': row[3] or 0,
            'accuracy': round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
        }
    return stats