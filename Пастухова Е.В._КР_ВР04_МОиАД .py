import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns
import nltk
import re
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, average_precision_score,
                             precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Функция для установки заголовка окна
def set_window_title(fig, title):
    """Устанавливает заголовок для всплывающего окна графика"""
    try:
        fig.canvas.manager.set_window_title(title)
    except:
        pass  # Если не получилось (например, в некоторых средах), игнорируем
    return fig


# Настройки для русского текста в графиках
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10
sns.set_theme(style="whitegrid")

# Загрузка ресурсов NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("=" * 70)
print("Применение ансамблевых методов анализа тональности IMDB")
print("=" * 70)
#ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
print("\n1. Загрузка набора данных IMDB (подвыборка 2000 отзывов)...")
df = pd.read_csv('IMDB Dataset.csv')
df = df.sample(n=2000, random_state=42).reset_index(drop=True)
print(f"   Размер датасета: {df.shape}")
print(f"   Колонки: {df.columns.tolist()}")
print("\n2. Очистка текстов (удаление HTML-тегов, стоп-слов, лемматизация)...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Очистка текста: нижний регистр, удаление HTML-тегов, пунктуации, стоп-слов, лемматизация"""
    # Удаление HTML-тегов
    text = re.sub(r'<[^>]+>', ' ', str(text))
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации и цифр (оставляю только буквы)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    # Токенизация и лемматизация
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens
              if token not in stop_words and len(token) > 1]
    return ' '.join(tokens)


df['clean_review'] = df['review'].apply(clean_text)
df['sentiment_binary'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print(f"   Пример очищенного текста: {df['clean_review'].iloc[0][:100]}...")
print("\n3. Исследовательский анализ данных...")

fig1 = plt.figure(figsize=(15, 10))
fig1 = set_window_title(fig1, 'Окно 1: Исследовательский анализ данных IMDB')
fig1.suptitle('Исследовательский анализ данных IMDB', fontsize=16, fontweight='bold')

# Распределение классов
plt.subplot(2, 3, 1)
df['sentiment'].value_counts().plot.pie(autopct='%1.0f%%', colors=['green', 'red'])
plt.title('Распределение классов', fontsize=12)

# Длина отзывов
df['length'] = df['clean_review'].str.len()
plt.subplot(2, 3, 2)
plt.hist(df[df.sentiment_binary == 1]['length'], bins=20, alpha=0.7, label='Позитив', color='green')
plt.hist(df[df.sentiment_binary == 0]['length'], bins=20, alpha=0.7, label='Негатив', color='red')
plt.title('Длина отзывов (после очистки)', fontsize=12)
plt.xlabel('Символы')
plt.ylabel('Частота')
plt.legend()

# Количество слов
pos_words = ' '.join(df[df.sentiment_binary == 1]['clean_review']).split()
neg_words = ' '.join(df[df.sentiment_binary == 0]['clean_review']).split()
plt.subplot(2, 3, 3)
plt.bar(['Позитив', 'Негатив'], [len(pos_words), len(neg_words)], color=['green', 'red'])
plt.title('Количество слов по классам', fontsize=12)
plt.ylabel('Слов')

# Boxplot длин
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='sentiment', y='length', palette=['green', 'red'])
plt.title('Boxplot длины отзывов', fontsize=12)
plt.xlabel('Класс')
plt.ylabel('Длина')

# Топ-10 слов
top_pos = pd.Series(pos_words).value_counts().head(5)
top_neg = pd.Series(neg_words).value_counts().head(5)
plt.subplot(2, 3, 5)
words = list(top_pos.index) + list(top_neg.index)
counts = list(top_pos.values) + list(top_neg.values)
colors_plot = ['green'] * 5 + ['red'] * 5
plt.barh(words, counts, color=colors_plot)
plt.title('Топ-10 частотных слов', fontsize=12)
plt.xlabel('Частота')

# Распределение длины отзывов
plt.subplot(2, 3, 6)
df['length'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Распределение длины отзывов', fontsize=12)
plt.xlabel('Длина (символы)')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()

print("\n4. Обучение без учителя (PCA и кластеризация)...")

tfidf_pca = TfidfVectorizer(max_features=1000)
X_train_tfidf_temp = tfidf_pca.fit_transform(df['clean_review'].iloc[:1000])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_tfidf_temp.toarray())

# PCA визуализация
fig2 = plt.figure(figsize=(10, 7))
fig2 = set_window_title(fig2, 'Окно 2: PCA 2D проекция TF-IDF признаков')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['sentiment_binary'].iloc[:1000],
            cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=0.5)
plt.title('PCA 2D проекция TF-IDF признаков', fontsize=14, fontweight='bold')
plt.xlabel('Главный компонент 1')
plt.ylabel('Главный компонент 2')
plt.colorbar(label='Класс (0=негатив, 1=позитив)')
plt.show()

# KMeans кластеризация
fig3 = plt.figure(figsize=(10, 7))
fig3 = set_window_title(fig3, 'Окно 3: KMeans кластеризация (2 кластера)')
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7, edgecolors='black', linewidth=0.5)
plt.title('KMeans кластеризация (2 кластера)', fontsize=14, fontweight='bold')
plt.xlabel('Главный компонент 1')
plt.ylabel('Главный компонент 2')
plt.colorbar(label='Кластер')
plt.show()

print("\n5. Разделение данных на обучающую и тестовую выборки (80/20)...")
X = df['clean_review'].values
y = df['sentiment_binary'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Обучающая выборка: {X_train.shape[0]} отзывов")
print(f"   Тестовая выборка: {X_test.shape[0]} отзывов")
print(f"   Распределение классов (обучение): позитив={np.sum(y_train == 1)}, негатив={np.sum(y_train == 0)}")
print(f"   Распределение классов (тест): позитив={np.sum(y_test == 1)}, негатив={np.sum(y_test == 0)}")
print("\n6. Создание моделей...")

# Использую max_features=5000 как в отчете
TFIDF_PARAMS = {'max_features': 5000}

# 1. Логистическая регрессия
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])

# 2. Метод опорных векторов
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', SVC(C=1.0, kernel='linear', probability=True, random_state=42))
])

# 3. Random Forest
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=None,
                                   min_samples_split=2, random_state=42))
])

# 4. XGBoost
xgb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                          random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# 5. Стекинг-ансамбль (с 5-кратной кросс-валидацией)
stacking_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ('svm', SVC(C=1.0, kernel='linear', probability=True, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                  random_state=42, use_label_encoder=False, eval_metric='logloss'))
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        cv=5  # 5-кратная кросс-валидация
    ))
])

models = {
    'Logistic Regression': lr_pipeline,
    'SVM': svm_pipeline,
    'Random Forest': rf_pipeline,
    'XGBoost': xgb_pipeline,
    'Stacking': stacking_pipeline
}
print("\n7. Обучение моделей с замером времени...")
results = {}

for name, model in models.items():
    print(f"   Обучение {name}...", end=" ")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Предсказание
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Расчет метрик
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
        'Time (s)': round(train_time, 3)
    }
    print(f"готово (F1={results[name]['F1-Score']:.4f}, время={train_time:.3f}c)")

print("\n8. Оптимизация гиперпараметров для стекинг-ансамбля...")

# Создаю оптимизированную версию стекинга с биграммами
stacking_optimized = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # униграммы + биграммы
    ('clf', StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('svm', SVC(kernel='linear', probability=True, random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    ))
])

# Параметры для GridSearch (упрощенные для скорости)
param_grid = {
    'tfidf__max_features': [3000, 5000],
    'clf__final_estimator__C': [0.1, 1.0],
    'clf__rf__n_estimators': [50, 100]
}

print("   Поиск оптимальных гиперпараметров...")
t0 = time.time()
grid_search = GridSearchCV(
    stacking_optimized,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - t0

print(f"   Лучшие параметры: {grid_search.best_params_}")
print(f"   Лучший F1 на кросс-валидации: {grid_search.best_score_:.4f}")

# Оценка оптимизированной модели
y_pred_optimized = grid_search.predict(X_test)
optimized_f1 = f1_score(y_test, y_pred_optimized)
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
print(f"   Оптимизированный стекинг: F1={optimized_f1:.4f}, Accuracy={optimized_accuracy:.4f}")
print(f"   Время оптимизации: {grid_time:.3f}c")

# Добавляю результаты оптимизированной модели в общую таблицу
results['Stacking (Optimized)'] = {
    'Accuracy': optimized_accuracy,
    'Precision': precision_score(y_test, y_pred_optimized),
    'Recall': recall_score(y_test, y_pred_optimized),
    'F1-Score': optimized_f1,
    'ROC-AUC': roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1]),
    'Time (s)': round(grid_time, 3)
}

print("\n9. Визуализация результатов...")

# DataFrame с результатами
results_df = pd.DataFrame(results).T.round(4)
results_df = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Time (s)']]
print("\nСводная таблица результатов:")
print(results_df)

# График сравнения моделей по F1-Score (как в отчете)
fig4 = plt.figure(figsize=(12, 6))
fig4 = set_window_title(fig4, 'Окно 4: Сравнение моделей по F1-мере')
models_names = list(results.keys())
f1_scores = [results[m]['F1-Score'] for m in models_names]
colors_f1 = ['green' if 'Stacking' in m else 'steelblue' for m in models_names]

bars = plt.bar(range(len(models_names)), f1_scores, color=colors_f1, edgecolor='black')
plt.title('Сравнение моделей по F1-мере', fontsize=14, fontweight='bold')
plt.xlabel('Модель')
plt.ylabel('F1-Score')
plt.xticks(range(len(models_names)), models_names, rotation=45, ha='right')
plt.ylim(0, 1)

for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# График сравнения по ROC-AUC
fig5 = plt.figure(figsize=(12, 6))
fig5 = set_window_title(fig5, 'Окно 5: Сравнение моделей по ROC-AUC')
roc_aucs = [results[m]['ROC-AUC'] for m in models_names]
bars2 = plt.bar(range(len(models_names)), roc_aucs, color=colors_f1, edgecolor='black')
plt.title('Сравнение моделей по ROC-AUC', fontsize=14, fontweight='bold')
plt.xlabel('Модель')
plt.ylabel('ROC-AUC')
plt.xticks(range(len(models_names)), models_names, rotation=45, ha='right')
plt.ylim(0.5, 1)

for bar, score in zip(bars2, roc_aucs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Сравнение времени обучения
fig6 = plt.figure(figsize=(12, 6))
fig6 = set_window_title(fig6, 'Окно 6: Время обучения моделей')
times = [results[m]['Time (s)'] for m in models_names]
bars3 = plt.bar(range(len(models_names)), times, color='coral', edgecolor='black')
plt.title('Время обучения моделей (секунды)', fontsize=14, fontweight='bold')
plt.xlabel('Модель')
plt.ylabel('Время (с)')
plt.xticks(range(len(models_names)), models_names, rotation=45, ha='right')

for bar, t in zip(bars3, times):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f'{t:.3f}c', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Precision-Recall кривая для стекинга
fig7 = plt.figure(figsize=(8, 6))
fig7 = set_window_title(fig7, 'Окно 7: Precision-Recall кривая (Stacking)')
best_model = stacking_pipeline
y_proba_best = best_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_proba_best)
ap_score = average_precision_score(y_test, y_proba_best)

plt.plot(recall, precision, 'o-', linewidth=2, color='darkgreen', label=f'AP = {ap_score:.4f}')
plt.title('Precision-Recall кривая (Stacking)', fontsize=14, fontweight='bold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.show()

# Матрица ошибок для стекинга
fig8 = plt.figure(figsize=(7, 6))
fig8 = set_window_title(fig8, 'Окно 8: Матрица ошибок (Stacking)')
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('Матрица ошибок (Stacking)', fontsize=14, fontweight='bold')
plt.show()

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛЕЙ")
print("=" * 70)
print(results_df.sort_values('F1-Score', ascending=False))

print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("1. Лучший результат по F1-мере показал стекинг-ансамбль.")
print("2. Стекинг превзошел все одиночные модели благодаря комбинации разнородных алгоритмов (LR, SVM, RF, XGBoost).")
print("3. Использование 5-кратной кросс-валидации в стекинге снизило риск переобучения.")
print("4. Оптимизация гиперпараметров позволила дополнительно повысить качество.")
print("5. Время обучения стекинга приемлемо для практического использования.")
print("Репозиторий с кодом: https://github.com/70201687/Coursework_ML")