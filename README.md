# Conversion Prediction Project

## Описание проекта
Модель машинного обучения для предсказания конверсии пользователей на основе данных о сессиях и событиях на сайте. Модель предсказывает вероятность совершения целевых действий пользователем.

### Целевые действия
- `sub_car_claim_click` - Клик по страховой претензии
- `sub_car_claim_submit_click` - Отправка страховой претензии
- `sub_open_dialog_click` - Открытие диалога
- `sub_custom_question_submit_click` - Отправка вопроса
- `sub_call_number_click` - Клик по номеру телефона
- `sub_callback_submit_click` - Запрос обратного звонка
- `sub_submit_success` - Успешная отправка формы
- `sub_car_request_submit_click` - Запрос автомобиля

## Структура проекта
```
project/
├── data/                          # Исходные данные
├── models/                        # Обученные модели
├── modules/                       # Исходный код
│   ├── main.py                   # FastAPI приложение
│   ├── main_notebook.ipynb       # Исследовательский анализ
│   ├── pipeline.py               # ML пайплайн обучения
│   └── run_api.py                # Скрипт запуска API
└── README.md                     # Документация
```

## Быстрый старт

### Вариант 1: Использование готовой модели (API)

1. **Установите зависимости:**
```bash
pip install fastapi uvicorn pydantic pandas numpy scikit-learn xgboost lightgbm
```

2. **Запустите API сервер:**
```bash
cd modules
python run_api.py
```

3. **API будет доступно по адресу:**
- Статус: http://localhost:8000/status
- Предсказание: POST http://localhost:8000/predict

### Вариант 2: Обучение модели с нуля

1. **Обучите модель:**
```bash
cd modules
python pipeline.py
```

2. **Модель сохранится в папку `models/`**

## Использование API

### Пример запроса предсказания:

```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "session_id": "test_001",
    "client_id": "client_001",
    "features": {
      "utm_medium": "organic",
      "device_category": "mobile",
      "device_os": "Android",
      "device_brand": "Samsung",
      "device_browser": "Chrome",
      "geo_country": "Russia",
      "geo_city": "Moscow",
      "visit_number": 1,
      "visit_hour": 14,
      "visit_dow": 1,
      "visit_is_weekend": 0,
      "visit_is_work_hour": 1,
      "device_screen_width": 360.0,
      "device_screen_height": 640.0,
      "device_screen_diag": 734.0,
      "device_screen_area": 230400.0,
      "device_screen_ratio": 0.5625,
      "utm_source_freq": 15000,
      "utm_campaign_freq": 5000,
      "utm_adcontent_freq": 2000,
      "utm_comb_freq": 1000,
      "geo_city_freq": 50000,
      "client_total_visits": 1,
      "utm_is_organic": 1,
      "utm_is_paid": 0,
      "utm_is_social": 0,
      "geo_is_top_city": 1,
      "is_first_visit": 1
    }
  }'
```

### Пример ответа:
```json
{
  "session_id": "test_001",
  "client_id": "client_001",
  "conversion_probability": 0.023,
  "conversion_prediction": 0,
  "prediction_threshold": 0.5
}
```
