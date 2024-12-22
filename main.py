import asyncio
import time
from datetime import datetime
import httpx
import requests
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import os
import plotly.graph_objects as go


# Сопоставление месяцев с сезонами
month_to_season = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}

# Реальные средние температуры (примерные данные) для городов по сезонам
seasonal_temperatures = {
    "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
    "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
    "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
    "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
    "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
    "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
    "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
    "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
    "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
    "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
    "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
    "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
    "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
    "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
    "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},
}


# Генерация данных о температуре
def generate_realistic_temperature_data(cities, num_years=10):
    dates = pd.date_range(start="2010-01-01", periods=365 * num_years, freq="D")
    data = []

    for city in cities:
        for date in dates:
            season = month_to_season[date.month]
            mean_temp = seasonal_temperatures[city][season]
            # Добавляем случайное отклонение
            temperature = np.random.normal(loc=mean_temp, scale=5)
            data.append({"city": city, "timestamp": date, "temperature": temperature})

    df = pd.DataFrame(data)
    df['season'] = df['timestamp'].dt.month.map(lambda x: month_to_season[x])
    return df


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=["timestamp"])


def get_trend_direction(city_data):
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])

    X, y = np.arange(len(city_data)).reshape(-1, 1), city_data['temperature']
    model.fit(X, y)
    y_predict = model.predict(X)
    return y_predict, "Uptrend" if model.named_steps['regression'].coef_[0] > 0 else "Downtrend"


def process_data(city_name, city_data):
    rolling_window = 30
    city_data['rolling_mean'] = city_data['temperature'].rolling(window=rolling_window).mean()
    city_data['rolling_std'] = city_data['temperature'].rolling(window=rolling_window).std()
    city_data['is_anomaly'] = np.abs(city_data['temperature'] - city_data['rolling_mean']) > (2 * city_data['rolling_std'])

    season_profile = city_data.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()

    y_predict, trend_direction = get_trend_direction(city_data)

    avg_temp = city_data['temperature'].mean()
    min_temp = city_data['temperature'].min()
    max_temp = city_data['temperature'].max()

    return {
        'df': city_data,
        'city': city_name,
        'average_temp': avg_temp,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'season_profile': season_profile,
        'trend_direction': trend_direction,
        'y_predict': y_predict
    }

def temperature_is_abnormal(season_profile, current_temp):
    current_season = month_to_season[datetime.now().month]
    season_data = season_profile.loc[season_profile['season'] == current_season]
    mean_temp = season_data['mean'].values[0]
    std_temp = season_data['std'].values[0]

    if mean_temp - 2 * std_temp <= current_temp <= mean_temp + 2 * std_temp:
        st.success("Текущая температура в норме.")
    else:
        st.warning("Температура аномальная!")

async def compare_sync_and_async(city, api_key):
    st.title("Сравнение синхронного и асинхронного API-запросов")

    num_requests = st.number_input("Количество запросов", min_value=1, max_value=10, value=3)

    if st.button("Сравнить запросы"):
        start_time = time.time()
        for _ in range(num_requests):
            get_current_temperature(city, api_key)
        sync_time = time.time() - start_time

        start_time = time.time()
        await asyncio.gather(*(get_temperature_async(city, api_key) for _ in range(num_requests)))
        async_time = time.time() - start_time

        st.write(f"Время выполнения синхронно для {num_requests} запросов: {sync_time:.2f} секунд")
        st.write(f"Время выполнения асинхронно для {num_requests} запросов: {async_time:.2f} секунд")
def main():
    st.title("Анализ температуры по городам")

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите файл с данными (CSV)", type="csv")
    if uploaded_file is None:
        if not os.path.exists("temperature_data.csv"):
            data = generate_realistic_temperature_data(list(seasonal_temperatures.keys()))
            data.to_csv('temperature_data.csv', index=False)
        else:
            data = load_data("temperature_data.csv")
    else:
        data = load_data(uploaded_file)

    st.sidebar.header("Параллельная обработка")
    n_jobs = st.sidebar.slider("Количество потоков", 1, 8, 1)

    start = time.time()
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_data)(city, info) for city, info in data.groupby('city')
    )
    end = time.time()
    #Время обработки практически не изменяется при изменении числа потоков, при работе с большими файлами оно уменьшается при увеличении числа потоков
    st.write(f"Параллельная обработка выполнена за {end - start:.2f} секунд")

    selected_city = st.sidebar.selectbox("Выберите город", [result['city'] for result in results])
    city_result = [res for res in results if res['city'] == selected_city][0]

    st.subheader(f"Данные для города {selected_city}")
    st.metric("Средняя температура", f"{city_result['average_temp']:.2f}")
    st.metric("Минимальная температура", f"{city_result['min_temp']:.2f}")
    st.metric("Максимальная температура", f"{city_result['max_temp']:.2f}")

    df = city_result["df"]
    city_data = df[df["city"] == selected_city]

    st.subheader(f"Анализ температур для {selected_city}")
    normal_values = city_data[~city_data["is_anomaly"]]
    anomalies = city_data[city_data["is_anomaly"]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=normal_values["timestamp"],
            y=normal_values["temperature"],
            mode='markers',
            name='Нормальные значения',
            opacity=0.7
        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["temperature"],
            mode='markers',
            marker=dict(color='red'),
            name='Аномалии'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=city_data["timestamp"],
            y=city_result["y_predict"],
            mode='lines',
            line=dict(color='green'),
            name='Тренд'

        )
    )

    st.plotly_chart(fig)

    st.write(f"Тренд: {city_result["trend_direction"]}")

    st.subheader("Сезонный профиль")
    st.write(city_result['season_profile'])

    st.subheader("Мониторинг текущей температуры")
    api_key = st.text_input("Введите API ключ OpenWeatherMap")

    if api_key:
        if st.button("Получить текущую температуру"):
            current_temp = get_current_temperature(selected_city, api_key)
            if current_temp:
                st.write(f"Текущая температура в {selected_city}: {current_temp}°C")
                temperature_is_abnormal(city_result["season_profile"], current_temp)

    asyncio.run(compare_sync_and_async(selected_city, api_key))


def get_current_temperature(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    time.sleep(1)
    if response.status_code == 200:
        return response.json()["main"]["temp"]
    else:
        st.error(f"Ошибка API: {response.json().get('message')}")


async def get_temperature_async(city, api_key):
    async with httpx.AsyncClient() as client:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = await client.get(url)
        await asyncio.sleep(1)
        if response.status_code == 200:
            return response.json()["main"]["temp"]
        else:
            st.error(f"Ошибка API: {response.json().get('message')}")

if __name__ == "__main__":
    main()


