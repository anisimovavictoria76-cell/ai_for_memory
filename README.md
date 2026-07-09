#  Тренажер памяти с нейронной сетью

**Интеллектуальное веб-приложение для изучения английских слов.**  
Проект содержит full-stack разработку и машинное обучение. Нейросеть анализирует ваши ошибки, предсказывает сложные слова и составляет персональный план обучения.

---

##  Скриншоты

| Главная страница | Тренажер | Рекомендации |
| :---: | :---: | :---: |
| <img width="1895" height="892" alt="image" src="https://github.com/user-attachments/assets/d971da4f-3707-4342-941c-2cb05cb0c6a1" />
 | <img width="1891" height="898" alt="image" src="https://github.com/user-attachments/assets/be657d40-7be4-41a8-9942-7a40859f4f33" />
 | <img width="852" height="566" alt="image" src="https://github.com/user-attachments/assets/348e349f-017e-41e4-b690-e739a7040cc7" />
 |

---

##  Стек технологий

| Категория | Технологии |
| :--- | :--- |
| **Backend** | Python 3.13, Flask |
| **Machine Learning** | scikit-learn (MLPClassifier), NumPy, Pandas |
| **База данных** | SQLite (с ORM через Pandas) |
| **Frontend** | HTML5, CSS3 (Dark Glassmorphism), JavaScript (Chart.js) |
| **Тестирование** | Pytest, Coverage |
| **DevOps** | Git, GitHub, Виртуальное окружение (venv) |

---

##  Архитектура нейросети (MLP)

Для предсказания вероятности ошибки была разработана **полносвязная нейросеть (Multilayer Perceptron)**. 

###  Характеристики модели

- **Тип:** MLPClassifier (бинарная классификация)
- **Библиотека:** scikit-learn
- **Задача:** Предсказать вероятность того, что пользователь ошибется при переводе слова.
- **Архитектура:**
  - Входной слой: 7 нейронов (признаки слова).
  - Скрытый слой 1: 24 нейрона (ReLU).
  - Скрытый слой 2: 48 нейронов (ReLU).
  - Скрытый слой 3: 24 нейрона (ReLU).
  - Выходной слой: 1 нейрон (Sigmoid для вероятности).

###  Входные признаки (Feature Engineering)

Модель принимает на вход **7 числовых признаков** для анализа сложности слова:

| Признак | Описание |
| :--- | :--- |
| **Difficulty** | Базовая сложность (от 1 до 3, задается пользователем). |
| **Attempts** | Количество попыток запомнить слово. |
| **Word Length** | Длина слова (количество букв). |
| **Unique Letters** | Количество уникальных букв (чем больше, тем сложнее). |
| **Vowels** | Количество гласных. |
| **Consonants** | Количество согласных. |
| **Syllables** | Количество слогов (влияет на фонетическую сложность). |

# 1. Клонируйте репозиторий
git clone https://github.com/ваш-логин/memory_trainer.git

cd memory_trainer

# 2. Создайте виртуальное окружение и установите зависимости
python -m venv .venv

source .venv/bin/activate  # Для Windows: .venv\Scripts\activate

pip install -r requirements.txt

# 3. Подготовьте базу данных и обучите нейросеть
python -c "import database; database.init_db()"

python train_model.py

# 4. Запустите приложение
python app.py

После запуска откройте браузер.
