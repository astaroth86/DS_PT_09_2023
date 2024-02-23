import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor

import pickle

def aplicar_mapeos(df:pd.DataFrame):
    mapeo_motores = {'Gasolina':1, 'Diésel':0, 'Híbrido enchufable':5, 'Eléctrico':6, 'Gas natural (CNG)':2, 'Híbrido':4, 'Gas licuado (GLP)':3}
    mapeo_transmision = {'Manual': 0, 'Automático': 1}

    df = df.replace(mapeo_motores)
    df = df.replace(mapeo_transmision)

    return df

def preparar_dataframe(df:pd.DataFrame):
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
    train = preparar_dataframe(train)
    test = preparar_dataframe(test)

    train, test = apply_onehot_encoder(train, ['make', "model", "color", "version"], test)

    test.to_csv("../data/processed/test_prepared.csv")

    X_train = train.drop(columns=["price"])
    y_train = train[["price"]]

    etr = ExtraTreesRegressor(n_estimators=500, max_depth= 100, n_jobs= 20, random_state=42)

    etr.fit(X_train, y_train)

    # Guarda el modelo en un archivo pickle
    with open('../model/extra_tree_model.pkl', 'wb') as f:
        pickle.dump(etr, f)
