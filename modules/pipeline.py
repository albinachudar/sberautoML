import os
import pickle
import time
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class DataPreparer:
    """Класс для подготовки данных"""

    def __init__(self, target_actions: List[str]):
        self.target_actions = target_actions

    def prepare_data(self, sessions_path: str, hits_path: str) -> pd.DataFrame:
        """Подготовка и объединение данных"""
        print('Загрузка и подготовка данных')

        # Загрузка данных
        df_sessions = pd.read_pickle(sessions_path)
        df_hits = pd.read_parquet(hits_path)

        # Копируем DataFrame для безопасности
        df_processed = df_sessions.copy()

        # Удаляем ненужные колонки с большим количеством пропусков
        df_processed = df_processed.drop(columns=['device_model', 'utm_keyword'], axis=1)

        # Удаляем строки с пропусками в utm_source (их мало)
        df_processed = df_processed.dropna(subset=['utm_source'])

        # Заполнение пропусков
        df_processed['device_brand'] = df_processed['device_brand'].fillna('other')
        df_processed['device_os'] = df_processed['device_os'].fillna('other')
        df_processed['utm_campaign'] = df_processed['utm_campaign'].fillna('other')
        df_processed['utm_adcontent'] = df_processed['utm_adcontent'].fillna('other')

        # Объединение даты и времени
        df_processed['visit_datetime'] = pd.to_datetime(
            df_processed['visit_date'].astype(str) + ' ' + df_processed['visit_time'].astype(str),
            utc=True,
            errors='coerce'
        )

        df_processed = df_processed.drop(
            columns=['visit_date', 'visit_time'],
            axis=1
        )

        # Создание целевой переменной
        df_hits['is_target'] = df_hits['event_action'].isin(self.target_actions)
        session_targets = df_hits.groupby('session_id')['is_target'].any().astype(int)

        # Объединение с сессиями
        df_processed['target'] = df_processed['session_id'].map(session_targets).fillna(0).astype(int)

        return df_processed


class FeatureEngineer:
    """Класс для создания новых признаков"""

    def __init__(self, top_cities: Optional[set] = None, social_sources: Optional[set] = None):
        self.top_cities = top_cities or {'Moscow', 'Saint Petersburg', 'Yekaterinburg', 'Krasnodar', 'Kazan'}
        self.social_sources = social_sources or {
            'QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
            'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
        }

    def parse_screen_resolution(self, res: Any) -> Tuple[float, float]:
        """Парсинг разрешения экрана"""
        if pd.isna(res):
            return np.nan, np.nan
        try:
            w, h = str(res).split('x')
            return float(w), float(h)
        except (ValueError, AttributeError):
            return np.nan, np.nan

    def remove_wh_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление выбросов в ширине и высоте экрана"""
        df = df.copy()

        # Защита от деления на 0
        height_no_zero = df['device_screen_height'].replace(0, np.nan)

        bad_wh = (
                (df['device_screen_width'] < 200) |
                (df['device_screen_width'] > 6000) |
                (df['device_screen_height'] < 200) |
                (df['device_screen_height'] > 4500) |
                (df['device_screen_width'] / height_no_zero.fillna(1) < 0.5) |
                (df['device_screen_width'] / height_no_zero.fillna(1) > 5.5)
        )

        df.loc[bad_wh, ['device_screen_width', 'device_screen_height']] = np.nan

        # Заполняем медианными значениями
        width_median = float(df['device_screen_width'].median())
        height_median = float(df['device_screen_height'].median())

        df['device_screen_width'] = df['device_screen_width'].astype('float64')
        df['device_screen_height'] = df['device_screen_height'].astype('float64')

        df['device_screen_width'].fillna(width_median, inplace=True)
        df['device_screen_height'].fillna(height_median, inplace=True)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание новых признаков"""
        print('Создание новых признаков')

        # Копируем DataFrame для безопасности
        df_processed = df.copy()

        # Создание временных признаков
        df_processed['visit_hour'] = df_processed['visit_datetime'].dt.hour
        df_processed['visit_dow'] = df_processed['visit_datetime'].dt.dayofweek
        df_processed['visit_is_weekend'] = (df_processed['visit_dow'] >= 5).astype(int)
        df_processed['visit_is_work_hour'] = ((df_processed['visit_hour'] >= 9) &
                                              (df_processed['visit_hour'] <= 18)).astype(int)

        # Разрешение экрана
        width_height = df_processed['device_screen_resolution'].apply(self.parse_screen_resolution)
        df_processed['device_screen_width'] = [x[0] for x in width_height]
        df_processed['device_screen_height'] = [x[1] for x in width_height]

        df_processed = self.remove_wh_outliers(df_processed)

        # Признаки устройства
        df_processed['device_screen_diag'] = np.sqrt(
            df_processed['device_screen_width'] ** 2 + df_processed['device_screen_height'] ** 2
        )
        df_processed['device_screen_area'] = df_processed['device_screen_width'] * df_processed['device_screen_height']

        # Защита от деления на 0
        df_processed['device_screen_ratio'] = df_processed['device_screen_width'] / df_processed[
            'device_screen_height'].replace(0, np.nan)
        df_processed['device_screen_ratio'].fillna(df_processed['device_screen_ratio'].median(), inplace=True)

        # UTM признаки
        df_processed['utm_is_organic'] = df_processed['utm_medium'].isin(['organic', 'referral', '(none)']).astype(int)
        df_processed['utm_is_paid'] = 1 - df_processed['utm_is_organic']

        df_processed['utm_source_freq'] = df_processed.groupby('utm_source')['session_id'].transform('count')
        df_processed['utm_comb_freq'] = df_processed.groupby(['utm_source', 'utm_medium', 'utm_campaign'])[
            'session_id'].transform('count')

        df_processed['utm_is_social'] = df_processed['utm_source'].isin(self.social_sources).astype(int)

        # Частотные признаки
        for col in ['utm_source', 'utm_campaign', 'utm_adcontent']:
            if col in df_processed.columns:
                freq = df_processed[col].value_counts()
                df_processed[f'{col}_freq'] = df_processed[col].map(freq).fillna(0).astype(int)

        # Гео признаки
        df_processed['geo_is_top_city'] = df_processed['geo_city'].isin(self.top_cities).astype(int)
        df_processed['geo_city_freq'] = df_processed.groupby('geo_city')['session_id'].transform('count')

        # Визиты
        df_processed['is_first_visit'] = (df_processed['visit_number'] == 1).astype(int)
        client_total_visits = df_processed.groupby('client_id')['visit_number'].max()
        df_processed['client_total_visits'] = df_processed['client_id'].map(client_total_visits)

        # Удаляем исходные колонки
        cols_to_drop = ['visit_datetime', 'device_screen_resolution',
                        'session_id', 'client_id', 'utm_source',
                        'utm_campaign', 'utm_adcontent']

        df_processed = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns])

        # Заполнение пропусков для категориальных признаков
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = df_processed[col].fillna('other')
            df_processed[col] = df_processed[col].replace('(not set)', 'other')

        # Заполнение пропусков для числовых признаков
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())

        return df_processed


class ModelPipelineBuilder:
    """Класс для построения ML пайплайнов"""

    def __init__(self):
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='other')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32))
        ])

    def build_preprocessor(self, numeric_cols: List[str], categorical_cols: List[str],
                           binary_cols: List[str]) -> ColumnTransformer:
        """Построение препроцессора"""
        categorical_transformer = self.categorical_transformer

        return ColumnTransformer(transformers=[
            ('numeric', self.numeric_transformer, numeric_cols),
            ('categorical', categorical_transformer, categorical_cols),
            ('binary', 'passthrough', binary_cols)
        ])

    def build_model_pipelines(self, X_train: pd.DataFrame, y_train: pd.Series,
                              numeric_cols: List[str], categorical_cols: List[str],
                              binary_cols: List[str]) -> Dict[str, Pipeline]:
        """Построение пайплайнов для разных моделей"""
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        # Для всех моделей используем dense препроцессор для консистентности
        # и чтобы избежать проблем с LightGBM
        preprocessor = self.build_preprocessor(numeric_cols, categorical_cols, binary_cols)

        models = {
            'LogisticRegression': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ]),
            'HistGradientBoosting': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', HistGradientBoostingClassifier(
                    random_state=42,
                    max_iter=100,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    class_weight='balanced',
                    early_stopping=True,
                    validation_fraction=0.1
                ))
            ]),
            'XGBoost': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', XGBClassifier(
                    random_state=42,
                    n_estimators=100,
                    learning_rate=0.03,
                    max_depth=5,
                    min_child_weight=10,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    scale_pos_weight=pos_weight,
                    eval_metric='auc',
                    n_jobs=-1,
                    tree_method='hist',
                    enable_categorical=False
                ))
            ]),
            'LightGBM': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LGBMClassifier(
                    random_state=42,
                    n_estimators=100,
                    learning_rate=0.03,
                    max_depth=5,
                    num_leaves=31,
                    min_child_samples=50,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    scale_pos_weight=pos_weight,
                    n_jobs=-1,
                    verbose=-1,
                    reg_alpha=0.1,
                    reg_lambda=0.1
                ))
            ]),
            'MLP': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(50, 25),
                    activation='relu',
                    solver='adam',
                    random_state=42,
                    max_iter=200,
                    early_stopping=True,
                    alpha=0.001
                ))
            ])
        }

        return models


class ModelTrainer:
    """Класс для обучения и оценки моделей"""

    def __init__(self, models: Dict[str, Pipeline]):
        self.models = models
        self.results = {}

    @staticmethod
    def process_categories(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           categorical_cols: List[str], top_n: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Обработка категориальных признаков"""
        train_processed = train_df.copy()
        test_processed = test_df.copy()

        for col in categorical_cols:
            top_categories = train_df[col].value_counts().head(top_n).index
            train_processed.loc[~train_processed[col].isin(top_categories), col] = 'other'
            test_processed.loc[~test_processed[col].isin(top_categories), col] = 'other'

        return train_processed, test_processed

    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           categorical_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Обучение и оценка моделей"""

        # Обработка категориальных признаков
        X_train_processed, X_test_processed = self.process_categories(
            X_train, X_test, categorical_cols, top_n=15
        )

        for name, model in self.models.items():
            print(f'\nОбучение {name}')
            model_start_time = time.time()

            try:
                # Обучаем модель
                model.fit(X_train_processed, y_train)
                train_time = time.time() - model_start_time

                # Предсказания
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)

                # Дополнительные метрики
                y_pred = model.predict(X_test_processed)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                self.results[name] = {
                    'name': name,
                    'model': model,
                    'auc': auc_score,
                    'accuracy': accuracy,
                    'f1': f1,
                    'train_time': train_time
                }

                print(f'  ROC-AUC: {auc_score:.4f}')
                print(f'  Accuracy: {accuracy:.4f}')
                print(f'  F1-Score: {f1:.4f}')
                print(f'  Время обучения: {train_time:.2f} сек')

            except Exception as e:
                print(f'  Ошибка: {e}')
                import traceback
                traceback.print_exc()
                self.results[name] = None

        return self.results

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Получение DataFrame для сравнения моделей"""
        comparison_data = []
        for name, result in self.results.items():
            if result is not None:
                comparison_data.append({
                    'Model': name,
                    'ROC-AUC': result['auc'],
                    'Accuracy': result['accuracy'],
                    'F1-Score': result['f1'],
                    'Train Time (s)': result['train_time'],
                    'Above Target': result['auc'] >= 0.65
                })

        return pd.DataFrame(comparison_data).sort_values('ROC-AUC', ascending=False)

    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """Получение лучшей модели"""
        successful_models = {k: v for k, v in self.results.items() if v is not None}
        best_model_name = max(successful_models, key=lambda x: successful_models[x]['auc'])
        return best_model_name, successful_models[best_model_name]


class ModelSaver:
    """Класс для сохранения моделей"""

    def __init__(self, output_dir: str = '../models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_model(self, model: Pipeline, model_name: str, model_info: Dict[str, Any],
                   X: pd.DataFrame, y: pd.Series, check_auc: float, training_time: float,
                   total_runtime: float) -> str:
        """Сохранение модели с метаинформацией"""

        model_package = {
            'model': model,
            'model_name': model_name,
            'model_type': type(model.named_steps['classifier']).__name__,
            'preprocessor': model.named_steps['preprocessor'] if 'preprocessor' in model.named_steps else None,
            'feature_names': X.columns.tolist(),
            'training_data_info': {
                'total_samples': len(X),
                'positive_samples': int(y.sum()),
                'negative_samples': int(len(y) - y.sum()),
                'conversion_rate': float(y.mean()),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'training_time_seconds': float(training_time),
                'total_runtime_seconds': float(total_runtime)
            },
            'performance_metrics': {
                'test_roc_auc': float(model_info['auc']),
                'test_accuracy': float(model_info['accuracy']),
                'test_f1_score': float(model_info['f1']),
                'validation_roc_auc': float(check_auc)
            },
            'model_parameters': model.named_steps['classifier'].get_params(),
            'pipeline_steps': list(model.named_steps.keys())
        }

        model_filename = f'{self.output_dir}/conversion_model_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model_package, f)

        return model_filename


def main():
    print('Conversion Prediction Pipeline')
    start_time = datetime.now()

    # Конфигурация
    TARGET_ACTIONS = [
        'sub_car_claim_click',
        'sub_car_claim_submit_click',
        'sub_open_dialog_click',
        'sub_custom_question_submit_click',
        'sub_call_number_click',
        'sub_callback_submit_click',
        'sub_submit_success',
        'sub_car_request_submit_click'
    ]

    FILEPATH_SESSIONS = '../data/ga_sessions.pkl'
    FILEPATH_HITS = '../data/ga_hits-002.parquet'

    # 1. Подготовка данных
    data_preparer = DataPreparer(TARGET_ACTIONS)
    df_raw = data_preparer.prepare_data(FILEPATH_SESSIONS, FILEPATH_HITS)

    # 2. Feature Engineering
    feature_engineer = FeatureEngineer()
    df_processed = feature_engineer.engineer_features(df_raw)

    # 3. Разделение данных
    X = df_processed.drop('target', axis=1)
    y = df_processed['target']

    print(f'Размер данных: {X.shape}')
    print(f'Конверсия: {y.mean():.2%}')

    # Определение типов признаков
    binary_cols = ['visit_is_weekend', 'visit_is_work_hour', 'utm_is_organic',
                   'utm_is_paid', 'utm_is_social', 'geo_is_top_city', 'is_first_visit']

    numeric_cols = X.select_dtypes(include=['int32', 'int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    numeric_cols = list(set(numeric_cols) - set(binary_cols))

    print(f'Числовых признаков: {len(numeric_cols)}')
    print(f'Категориальных признаков: {len(categorical_cols)}')
    print(f'Бинарных признаков: {len(binary_cols)}')
    print(f'Всего признаков: {X.shape[1]}')

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Построение пайплайнов моделей
    pipeline_builder = ModelPipelineBuilder()
    models = pipeline_builder.build_model_pipelines(X_train, y_train, numeric_cols, categorical_cols, binary_cols)

    # 5. Обучение и оценка моделей
    trainer = ModelTrainer(models)
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, categorical_cols)

    # 6. Сравнение моделей
    comparison_df = trainer.get_comparison_dataframe()

    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # 7. Выбор лучшей модели
    best_model_name, best_model_info = trainer.get_best_model()

    print(f'\nЛУЧШАЯ МОДЕЛЬ: {best_model_name}')
    print(f'   ROC-AUC: {best_model_info["auc"]:.4f}')

    # 8. Обучение лучшей модели на всех данных
    print(f'\nОбучение лучшей модели ({best_model_name}) на всех данных')

    # Обрабатываем категории для всего датасета
    X_full_processed, _ = trainer.process_categories(X, X, categorical_cols, top_n=15)

    # Обучение финальной модели
    final_model = models[best_model_name]

    training_start_time = time.time()
    final_model.fit(X_full_processed, y)
    training_time = time.time() - training_start_time

    print(f'Обучение завершено за {training_time:.2f} секунд')

    # 9. Валидация финальной модели
    X_val, X_check, y_val, y_check = train_test_split(
        X_full_processed, y,
        test_size=0.1,
        random_state=42,
        stratify=y
    )

    check_model = final_model
    check_model.fit(X_val, y_val)

    y_check_pred_proba = check_model.predict_proba(X_check)[:, 1]
    check_auc = roc_auc_score(y_check, y_check_pred_proba)
    check_accuracy = accuracy_score(y_check, check_model.predict(X_check))
    check_f1 = f1_score(y_check, check_model.predict(X_check))

    print(f'\nПроверка на валидационной выборке (10% данных):')
    print(f'  ROC-AUC: {check_auc:.4f}')
    print(f'  Accuracy: {check_accuracy:.4f}')
    print(f'  F1-Score: {check_f1:.4f}')
    print(f'  Целевое значение (0.65) достигнуто: {check_auc >= 0.65}')

    # 10. Сохранение модели
    total_runtime = (datetime.now() - start_time).total_seconds()
    model_saver = ModelSaver()
    model_filename = model_saver.save_model(
        final_model, best_model_name, best_model_info,
        X, y, check_auc, training_time, total_runtime
    )

    print(f'\nМодель сохранена: {model_filename}')


if __name__ == '__main__':
    main()