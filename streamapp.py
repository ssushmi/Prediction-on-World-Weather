# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from math import sqrt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

# Load your dataset
# Note: Replace 'your_dataset.csv' with the path to your dataset
@st.cache_data
def load_data():
    df = pd.read_csv('GlobalWeatherRepository.csv')
    return df

df2 = load_data()

# Title of your Streamlit application
st.title("Weather Data Analysis")

image_url = "down2.jpg"  
st.image(image_url, use_column_width=True)
# Sidebar for choosing analysis type
analysis_type = st.sidebar.selectbox("Choose the Analysis Type", [ "Temperature Forecast", "Air Quality Prediction", "Cloud Prediction"])
unique_countries = df2['country'].unique()  # Assuming 'country' is the column name
country_selected = st.sidebar.selectbox("Select a Country", unique_countries)
df_filtered = df2[df2['country'] == country_selected]


# Function for Temperature Forecast
def temperature_forecast(df2):
    data = df2.copy()
    data['last_updated'] = pd.to_datetime(data['last_updated'])
    data.set_index('last_updated', inplace=True)
    daily_temperature = data['temperature_celsius'].resample('D').mean()
    daily_temperature.dropna(inplace=True)

    # Check for stationarity
    adf_result = adfuller(daily_temperature)
    if adf_result[1] > 0.05:
        daily_temperature = daily_temperature.diff().dropna()

    # Split the data
    train_size = int(len(daily_temperature) * 0.8)
    train, test = daily_temperature[0:train_size], daily_temperature[train_size:]

    # Fit the ARIMA model
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=len(test))

    # Evaluate the model
    rmse = sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    # Print results
    #st.write('Test RMSE:', rmse)
    #st.write('Test MAE:', mae)

    forecast = model_fit.forecast(steps=len(test))
    forecast_dates = test.index
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Temperature': forecast})
    st.write(forecast_df)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index, train, label='Train')
    ax.plot(test.index, test, label='Test', color='black')
    ax.plot(test.index, forecast, label='Forecast', color='red')
    ax.set_title('Temperature Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (Celsius)')
    ax.legend()
    st.pyplot(fig)

    




# Function for Cloud Prediction
def cloud_prediction(df2):
    
    features_classification = df2[[
    'latitude', 'longitude', 'wind_kph', 'temperature_celsius', 'pressure_mb',
    'precip_mm', 'humidity', 'cloud', 'visibility_km', 'uv_index',
    'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_us-epa-index']]
    target_classification = df2['condition_text']

    # Split the data
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        features_classification, target_classification, test_size=0.2, random_state=42)

    # Create and fit the model
    rf_class_model = RandomForestClassifier(n_estimators=100)
    rf_class_model.fit(X_train_class, y_train_class)

    # Make predictions
    predictions_class = rf_class_model.predict(X_test_class)

    # Evaluate the model
    accuracy_class = accuracy_score(y_test_class, predictions_class)
    class_report_class = classification_report(y_test_class, predictions_class)

    # Print the results
    st.write(f'Accuracy (Random Forest Classification): {accuracy_class}')
    st.write('Classification Report (Random Forest Classification):')
    st.write(class_report_class)

    # Feature importance visualization
    feature_importance = rf_class_model.feature_importances_
    feature_names = features_classification.columns
    indices = np.argsort(feature_importance)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(feature_importance)), feature_importance[indices])
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names[indices], rotation=90)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title('Random Forest Classifier - Feature Importance')
    st.pyplot(fig)




# Function for Air Quality Prediction
def air_quality_prediction(df2):
    features = [
    'temperature_celsius', 
    'humidity', 
    'wind_kph', 
    'pressure_mb', 
    'precip_mm', 
    'cloud', 
    'visibility_km', 
    'uv_index', ]

    missing_data = df2[features].isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        for feature in missing_data.index:
            df2[feature].fillna(df2[feature].mean(), inplace=True)

    # Prepare the data
    X = df2[features]
    y = df2['air_quality_us-epa-index']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the SVM classifier
    classifier = svm.SVC()
    classifier.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #st.write(f'The accuracy of the SVM model on the test set is: {accuracy:.2f}')


    # Plotting actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predictions')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', lw=2, label='Perfect Prediction')
    ax.set_title('Actual vs Predicted Air Quality')
    ax.set_xlabel('Actual Air Quality Index')
    ax.set_ylabel('Predicted Air Quality Index')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


    y_pred = classifier.predict(X_test)
    # Combine actual and predicted values into a DataFrame
    comparison_df = pd.DataFrame({'Actual Air Quality Index': y_test, 'Predicted Air Quality Index': y_pred})
    st.write(comparison_df)


# Calling functions based on user selection

if analysis_type == "Temperature Forecast":
    temperature_forecast(df_filtered)
elif analysis_type == "Cloud Prediction":
    cloud_prediction(df_filtered)
elif analysis_type == "Air Quality Prediction":
    air_quality_prediction(df_filtered)


country_coords = df2.groupby('country')[['latitude', 'longitude']].mean().reset_index()

def create_map(country_selected):
    if country_selected in country_coords['country'].values:
        lat, lon = country_coords[country_coords['country'] == country_selected][['latitude', 'longitude']].values[0]
        m = folium.Map(location=[lat, lon], zoom_start=4)
        folium.Marker([lat, lon], tooltip=country_selected).add_to(m)
        folium_static(m)
    else:
        st.write("Country not found in dataset")
if country_selected:
    create_map(country_selected)