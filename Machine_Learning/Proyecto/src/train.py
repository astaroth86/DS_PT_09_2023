import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

import pickle

def aplicar_label_encoder(df:pd.DataFrame):
    le = LabelEncoder()
    # df["model"] = le.fit_transform(df["model"])
    # df["fuel"] = le.fit_transform(df["fuel"])
    # df["shift"] = le.fit_transform(df["shift"])
    # df["color"] = le.fit_transform(df["color"])
    df["version"] = le.fit_transform(df["version"])

    return df


def aplicar_mapeos(df:pd.DataFrame):
    mapeo_motores = {'Gasolina':1, 'Diésel':0, 'Híbrido enchufable':5, 'Eléctrico':6, 'Gas natural (CNG)':2, 'Híbrido':4, 'Gas licuado (GLP)':3}
    mapeo_transmision = {'Manual': 0, 'Automático': 1}

    df = df.replace(mapeo_motores)
    df = df.replace(mapeo_transmision)

    return df

def preparar_dataframe(df:pd.DataFrame):
    df = aplicar_label_encoder(df)
    # df = aplicar_one_hot_encoder(df)
    df = aplicar_mapeos(df)

    return df

def apply_onehot_encoder(train:pd.DataFrame, columns_to_encode:list, test:pd.DataFrame=None):
    
    # Resetear índices para evitar desalineación
    train = train.reset_index(drop=True)
    
    # Crear el OneHotEncoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Ajustar y transformar las columnas seleccionadas
    transformed_data = encoder.fit_transform(train[columns_to_encode])

    # Crear un DataFrame con las columnas transformadas
    transformed_df = pd.DataFrame(transformed_data, columns=encoder.get_feature_names_out(columns_to_encode))
    
    # Concatenar con el DataFrame original excluyendo las columnas transformadas
    df_concatenated = pd.concat([train.drop(columns_to_encode, axis=1), transformed_df], axis=1)

    # Si se proporciona un segundo DataFrame, aplicar la misma transformación
    if test is not None:
        transformed_data_to_transform = encoder.transform(test[columns_to_encode])
        transformed_df_to_transform = pd.DataFrame(transformed_data_to_transform, columns=encoder.get_feature_names_out(columns_to_encode))
        df_to_transform_concatenated = pd.concat([test.drop(columns_to_encode, axis=1), transformed_df_to_transform], axis=1)
        return df_concatenated, df_to_transform_concatenated

    return df_concatenated

def entrenar_modelo(train, test):
    train, test = apply_onehot_encoder(train, ['make', "model", "color"], test)

    test.to_csv("data/processed/test_prepared.csv")

    X_train = train.drop(columns=["price"])
    y_train = train[["price"]]

    # X_test = test.drop(columns=["price"])
    # y_test = test[["price"]]

    etr = ExtraTreesRegressor(n_estimators=500, max_depth= 100, n_jobs= 10, random_state=42)
    # rfr = RandomForestRegressor(n_estimators=500, max_depth= 100, n_jobs= -1, random_state=42)

    etr.fit(X_train, y_train)

    # Guarda el modelo en un archivo pickle
    with open('model/extra_tree_model.pkl', 'wb') as f:
        pickle.dump(etr, f)

    # rfr.fit(X_train, y_train)
    # y_pred_train_rfr = rfr.predict(X_train)

    # print(mean_squared_error(y_train, y_pred_train_rfr))
    
    # # Guarda el modelo en un archivo pickle
    # with open('model/random_forest_model.pkl', 'wb') as f:
    #     pickle.dump(rfr, f)

def cargar_modelo(test:pd.DataFrame):
    X_test = test.drop(columns=["price"])
    X_test = X_test.drop('Unnamed: 0.1', axis=1)
    
    y_test = test[["price"]]

    with open('model/extra_tree_model.pkl', 'rb') as f:

        modelo = pickle.load(f)
        y_pred_train_etr = modelo.predict(X_test)
        print("Extra tree")
        print("Coeficiente determinación", r2_score(y_test, y_pred_train_etr))
        print("MAE", mean_absolute_error(y_test, y_pred_train_etr))
        print("MAPE", mean_absolute_percentage_error(y_test, y_pred_train_etr))
        print("MSE", mean_squared_error(y_test, y_pred_train_etr))
        print("RMSE", np.sqrt(mean_squared_error(y_test, y_pred_train_etr)))


    # with open('model/random_forest_model.pkl', 'rb') as f:
        
    #     modelo = pickle.load(f)
    #     y_pred_train_rfr = modelo.predict(X_test)

    #     print("Random Forest")
    #     print("Coeficiente determinación", r2_score(y_test, y_pred_train_rfr))
    #     print("MAE", mean_absolute_error(y_test, y_pred_train_rfr))
    #     print("MAPE", mean_absolute_percentage_error(y_test, y_pred_train_rfr))
    #     print("MSE", mean_squared_error(y_test, y_pred_train_rfr))
    #     print("RMSE", np.sqrt(mean_squared_error(y_test, y_pred_train_rfr)))


train = pd.read_csv("data/processed/train.csv")

train = preparar_dataframe(train)

test = pd.read_csv("data/processed/test.csv")
test = preparar_dataframe(test)

entrenar_modelo(train, test)

cargar_modelo(pd.read_csv("data/processed/test_prepared.csv"))