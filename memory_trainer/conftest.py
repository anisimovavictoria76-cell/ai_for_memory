import sys
import os

# Добавка корневой папки проекта в PYTHONPATH
# Это позволяет pytest находить модули app, database, train_model
sys.path.insert(0, os.path.dirname(__file__))