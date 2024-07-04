import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# Función para cargar datos
@st.cache_data
def load_data():
    ruta1= "streamlit/data/df_modelo2.parquet"
    
    dfmodelo = pd.read_parquet(ruta1)
    
    return dfmodelo

# Cargar datos
dfmodelo = load_data()



# Convertir la columna review_date a formato numérico
dfmodelo['review_date'] = pd.to_datetime(dfmodelo['review_date']).dt.strftime('%Y%m%d').astype(int)

# Filtrar los datos para los restaurantes Taco Bell
taco_bell_data = dfmodelo[dfmodelo['business_name'].str.contains('Taco Bell', case=False)]

# Seleccionar las características y la variable objetivo
features = ['review_date', 'city']
target = 'business_rating'

# Codificar la columna 'city' utilizando one-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
city_encoded = encoder.fit_transform(taco_bell_data[['city']])

# Convertir el resultado de one-hot encoding a un DataFrame
city_encoded_df = pd.DataFrame(city_encoded.toarray(), columns=encoder.get_feature_names_out(['city']))

# Concatenar las características codificadas con las otras características
taco_bell_data_encoded = pd.concat([taco_bell_data[['review_date']].reset_index(drop=True), city_encoded_df], axis=1)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(taco_bell_data_encoded, taco_bell_data[target], test_size=0.2, random_state=42)

# Crear el modelo de regresión XGBoost
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

#####

st.title("Predicción de Ratings para Taco Bell")

# Interfaz para ingresar ciudad y año
st.header("Predicción de rating para una ciudad y año específicos")
ciudad = st.text_input("Ingresa la ciudad")
anio = st.number_input("Ingresa el año", min_value=2023, max_value=2030, step=1)

if st.button("Predecir"):
    if ciudad and anio:
        new_data = pd.DataFrame({'review_date': [int(f"{anio}0101")], 'city': [ciudad]})
        new_city_encoded = encoder.transform(new_data[['city']])
        new_city_encoded_df = pd.DataFrame(new_city_encoded.toarray(), columns=encoder.get_feature_names_out(['city']))

        new_data_encoded = pd.concat([new_data[['review_date']], new_city_encoded_df], axis=1)

        missing_cols = set(X_train.columns) - set(new_data_encoded.columns)
        for c in missing_cols:
            new_data_encoded[c] = 0
        new_data_encoded = new_data_encoded[X_train.columns]

        new_predictions = model.predict(new_data_encoded)

        st.write(f"Predicted business_rating for Taco Bell in {ciudad} in the year {anio}: {new_predictions[0]}")


