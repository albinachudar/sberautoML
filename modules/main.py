import glob
import pickle
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os


app = FastAPI(title='Conversion Prediction API',
              description='API для предсказания конверсии пользователей',
              version='1.0.0')


def load_latest_model():
    """Загрузка последней сохраненной модели"""
    try:
        model_files = glob.glob('../models/conversion_model_*.pkl')

        if not model_files:
            raise FileNotFoundError('Модели не найдены!')

        # Берем самый свежий файл
        latest_model = max(model_files, key=os.path.getctime)

        with open(latest_model, 'rb') as f:
            model_package = pickle.load(f)

        return model_package

    except Exception as e:
        print(f'Ошибка: {e}')
        return None


# Загрузка модели
model_package = load_latest_model()
if model_package:
    model = model_package['model']
else:
    raise RuntimeError("Не удалось загрузить модель!")


# Pydantic модели для валидации данных
class ConversionFeatures(BaseModel):
    """Модель для валидации входных признаков"""
    # Категориальные признаки
    utm_medium: str
    device_category: str
    device_os: str
    device_brand: str
    device_browser: str
    geo_country: str
    geo_city: str

    # Числовые признаки
    visit_number: int
    visit_hour: int
    visit_dow: int
    visit_is_weekend: int
    visit_is_work_hour: int
    device_screen_width: float
    device_screen_height: float
    device_screen_diag: float
    device_screen_area: float
    device_screen_ratio: float
    utm_source_freq: int
    utm_campaign_freq: int
    utm_adcontent_freq: int
    utm_comb_freq: int
    geo_city_freq: int
    client_total_visits: int
    utm_is_organic: int
    utm_is_paid: int
    utm_is_social: int
    geo_is_top_city: int
    is_first_visit: int


class ConversionRequest(BaseModel):
    """Модель для запроса предсказания"""
    session_id: str
    client_id: str
    features: ConversionFeatures


class ConversionResponse(BaseModel):
    """Модель для ответа с предсказанием"""
    session_id: str
    client_id: str
    conversion_probability: float
    conversion_prediction: int


class ModelInfo(BaseModel):
    """Модель для информации о модели"""
    model_name: str
    model_type: str
    training_date: str
    total_samples: int
    conversion_rate: float
    test_roc_auc: float
    test_accuracy: float
    feature_count: int


# Вспомогательные функции
def prepare_features_for_prediction(features_dict: dict) -> pd.DataFrame:
    """Подготовка признаков для предсказания"""
    # Создаем DataFrame с правильным порядком колонок
    if 'feature_names' in model_package:
        feature_names = model_package['feature_names']
        df = pd.DataFrame([features_dict])[feature_names]
    else:
        df = pd.DataFrame([features_dict])

    # Преобразуем категориальные признаки в строки
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    return df


# API endpoints
@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Conversion Prediction API",
        "version": "1.0.0",
        "status": "/status",
        "model_info": "/model_info"
    }


@app.get("/status")
async def status():
    """Проверка статуса API"""
    return {"status": "OK", "message": "API работает корректно"}


@app.get("/model_info", response_model=ModelInfo)
async def model_info():
    """Информация о загруженной модели"""
    return {
        "model_name": model_package['model_name'],
        "model_type": model_package['model_type'],
        "training_date": model_package['training_data_info']['training_date'],
        "total_samples": model_package['training_data_info']['total_samples'],
        "conversion_rate": model_package['training_data_info']['conversion_rate'],
        "test_roc_auc": model_package['performance_metrics']['test_roc_auc'],
        "test_accuracy": model_package['performance_metrics']['test_accuracy'],
        "feature_count": len(model_package['feature_names'])
    }


@app.post("/predict", response_model=ConversionResponse)
async def predict(request: ConversionRequest):
    """
    Предсказание вероятности конверсии

    - **session_id**: ID сессии пользователя
    - **client_id**: ID клиента
    - **features**: Признаки для предсказания
    """
    try:
        # Подготавливаем данные
        features_dict = request.features.dict()
        df = prepare_features_for_prediction(features_dict)

        # Делаем предсказание
        probabilities = model.predict_proba(df)[:, 1]
        probability = float(probabilities[0])

        # Бинарное предсказание (порог 0.5)
        prediction = 1 if probability >= 0.5 else 0

        return {
            "session_id": request.session_id,
            "client_id": request.client_id,
            "conversion_probability": probability,
            "conversion_prediction": prediction,
            "prediction_threshold": 0.5
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при предсказании: {str(e)}")


@app.post("/predict_batch", response_model=List[ConversionResponse])
async def predict_batch(requests: List[ConversionRequest]):
    """
    Пакетное предсказание для нескольких записей
    """
    try:
        results = []

        for request in requests:
            features_dict = request.features.dict()
            df = prepare_features_for_prediction(features_dict)

            probabilities = model.predict_proba(df)[:, 1]
            probability = float(probabilities[0])
            prediction = 1 if probability >= 0.5 else 0

            results.append({
                "session_id": request.session_id,
                "client_id": request.client_id,
                "conversion_probability": probability,
                "conversion_prediction": prediction,
                "prediction_threshold": 0.5
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при пакетном предсказании: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)