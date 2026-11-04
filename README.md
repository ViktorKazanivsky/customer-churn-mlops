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


## Валідація та версіонування даних
```
python src/validate_data.py
python src/version_data.py
python src/train.py
```
- `validate_data.py` перевіряє дані за схемою `configs/data_schema.yaml`. Статус **passed** означає, що дані коректні, **failed** — що знайдено помилки, і навчання запускати не можна.
- Звіт валідації: `reports/data_validation_report.json`.
- Версія датасету: `data/processed/churn_v1.csv`, метадані: `churn_v1_metadata.json`.
- Модель навчається на версії даних **v1**.

## Висновок
Базова модель має добру загальну якість (ROC-AUC 0.84), але виявляє лише 57% клієнтів, що йдуть. Вона є точкою відліку для подальших покращень.