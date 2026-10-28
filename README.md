# Customer Churn MLOps

## Мета
Прогнозування відтоку клієнтів телеком-компанії (бінарна класифікація).

## Дані
Telco Customer Churn (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Завантажити CSV, перейменувати на `churn.csv` і покласти в `data/raw/`.

## Структура
- `data/raw` — сирі дані
- `data/processed` — підготовлені дані
- `src` — код
- `models` — збережені моделі (не комітяться в Git)
- `reports` — метрики
- `notebooks` — експерименти
- `tests` — тести

## Встановлення
```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск навчання
```
python src/train.py
```

## Результати (LogisticRegression)
| Метрика | Значення |
|---|---|
| Accuracy | 0.807 |
| Precision | 0.658 |
| Recall | 0.567 |
| F1 | 0.609 |
| ROC-AUC | 0.842 |

## Висновок
Базова модель має добру загальну якість (ROC-AUC 0.84), але виявляє лише 57% клієнтів, що йдуть. Вона є точкою відліку для подальших покращень.