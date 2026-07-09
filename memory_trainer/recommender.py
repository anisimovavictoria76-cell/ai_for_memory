import database
import train_model
from datetime import datetime, timedelta
import random


class WordRecommender:
    """Рекомендательная система для изучения слов"""

    def __init__(self, session_id='default'):
        self.session_id = session_id
        self.all_words = database.get_all_words()
        self.stats = database.get_all_words_stats()

    def get_words_to_review(self, limit=10):
        """
        Возвращает слова, которые нужно повторить сегодня.
        Основано на частоте ошибок и времени последнего повторения.
        """
        words_to_review = []

        # Если нет статистики, возвращаем все слова
        if not self.stats:
            for word in self.all_words:
                words_to_review.append({
                    'word': word['word'],
                    'translation': word['translation'],
                    'reason': 'новое слово',
                    'priority': 'medium',
                    'score': 0.5
                })
            return words_to_review[:limit]

        for word in self.all_words:
            word_name = word['word']
            stat = self.stats.get(word_name)

            # Если статистики по слову нет
            if stat is None:
                words_to_review.append({
                    'word': word_name,
                    'translation': word['translation'],
                    'reason': 'новое слово',
                    'priority': 'medium',
                    'score': 0.5
                })
                continue

            # 1. Слова с ошибками (> 2 ошибок)
            if stat.get('wrong', 0) >= 2:
                words_to_review.append({
                    'word': word_name,
                    'translation': word['translation'],
                    'reason': f'много ошибок ({stat["wrong"]})',
                    'priority': 'high',
                    'score': stat.get('wrong', 0) * 2
                })

            # 2. Слова с низкой точностью (< 60%)
            elif stat.get('accuracy', 100) < 60 and stat.get('total', 0) > 0:
                words_to_review.append({
                    'word': word_name,
                    'translation': word['translation'],
                    'reason': f'точность {stat["accuracy"]}%',
                    'priority': 'high',
                    'score': (100 - stat['accuracy']) / 10
                })

            # 3. Слова, которые давно не повторялись
            elif stat.get('total', 0) == 0:
                words_to_review.append({
                    'word': word_name,
                    'translation': word['translation'],
                    'reason': 'новое слово',
                    'priority': 'medium',
                    'score': 0.5
                })

            # 4. Слова со средней точностью (60-80%)
            elif 60 <= stat.get('accuracy', 0) < 80:
                words_to_review.append({
                    'word': word_name,
                    'translation': word['translation'],
                    'reason': f'средняя точность {stat["accuracy"]}%',
                    'priority': 'medium',
                    'score': 1
                })

        # score
        words_to_review.sort(key=lambda x: x.get('score', 0), reverse=True)

        return words_to_review[:limit]

    def get_words_to_skip(self, limit=5):
        """
        Возвращает слова, которые можно пропустить (т.е уже хорошо выучены).
        """
        words_to_skip = []

        for word in self.all_words:
            word_name = word['word']
            stat = self.stats.get(word_name)

            if stat is None:
                continue

            if stat.get('accuracy', 0) >= 90 and stat.get('total', 0) >= 5:
                words_to_skip.append({
                    'word': word_name,
                    'translation': word['translation'],
                    'accuracy': stat.get('accuracy', 0),
                    'attempts': stat.get('total', 0),
                    'priority': 'low'
                })

        return words_to_skip[:limit]

    def get_weekly_plan(self):
        """
        Генерирует персональный план на неделю.
        """
        # Собираем слова для повторения
        review_words = self.get_words_to_review(limit=21)

        # Если слов мало, добавляем все слова
        if len(review_words) < 7:
            all_words_names = [w['word'] for w in self.all_words]
            for word in all_words_names:
                if word not in [r['word'] for r in review_words]:
                    translation = next((w['translation'] for w in self.all_words if w['word'] == word), '')
                    review_words.append({
                        'word': word,
                        'translation': translation,
                        'reason': 'добавлено в план',
                        'priority': 'medium',
                        'score': 0.5
                    })
                    if len(review_words) >= 21:
                        break

        # Разбиваем по дням
        days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
        plan = {}

        words_per_day = 3
        for i, day in enumerate(days):
            start_idx = i * words_per_day
            end_idx = start_idx + words_per_day
            plan[day] = review_words[start_idx:end_idx] if start_idx < len(review_words) else []

        # Статистика
        total_words = len(self.all_words)
        total_learned = 0
        total_correct = 0
        total_wrong = 0

        for stat in self.stats.values():
            if stat.get('total', 0) > 0:
                total_learned += 1
            total_correct += stat.get('correct', 0)
            total_wrong += stat.get('wrong', 0)

        total_attempts = total_correct + total_wrong
        overall_accuracy = round(total_correct / total_attempts * 100, 1) if total_attempts > 0 else 0

        return {
            'plan': plan,
            'summary': {
                'total_words': total_words,
                'learned_words': total_learned,
                'total_correct': total_correct,
                'total_wrong': total_wrong,
                'overall_accuracy': overall_accuracy
            }
        }

    def get_recommendation_for_session(self, limit=10):
        """
        Возвращает рекомендации для текущей сессии обучения.
        """
        to_review = self.get_words_to_review(limit=limit)

        result = []
        for word_data in to_review:
            word = word_data['word']
            difficulty = next((w['difficulty'] for w in self.all_words if w['word'] == word), 1)
            stat = self.stats.get(word, {})
            attempts = stat.get('total', 0) + 1

            # Предсказание вероятности ошибки
            try:
                forget_prob = train_model.predict_forget_probability(word, difficulty, attempts)
            except:
                forget_prob = 0.5

            word_data['forget_probability'] = round(forget_prob * 100, 1)
            result.append(word_data)

        return result