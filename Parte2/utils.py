# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def get_data():
    GSPREADHSEET_DOWNLOAD_URL = (
        "https://docs.google.com/spreadsheets/d/{gid}/export?format=csv&id={gid}".format
    )
    TP_GID = '1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0'
    df = pd.read_csv(GSPREADHSEET_DOWNLOAD_URL(gid=TP_GID))
    return feature_engineering(df)

def get_holdout_data():
    GSPREADHSEET_DOWNLOAD_URL = (
        "https://docs.google.com/spreadsheets/d/{gid}/export?format=csv&id={gid}".format
    )
    TP_GID = '1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE'
    df = pd.read_csv(GSPREADHSEET_DOWNLOAD_URL(gid=TP_GID))
    del df['representatividad_poblacional']
    ids = df['id'].copy()
    del df['id']
    return ids, feature_engineering(df)

def escribir_holdout(predicciones : np.array, nombre, ids):
    archivo = open("Holdout/" + nombre + ".csv", "w")
    archivo.write("id , tiene_alto_valor_adquisitivo\n")
    i = 0
    for prediccion in predicciones:
        archivo.write(str(ids[i])+ "," + str(prediccion) + "\n")
        i = i + 1
    archivo.close()


def feature_engineering(df):
    # Missings en barrio
    data_set_mejorado = df.copy()
    data_set_mejorado['barrio'] = data_set_mejorado['barrio'].apply(lambda x: 'Palermo' if str(x) == 'nan' else x)

    # Missings en categoría de trabajo
    data_set_mejorado['categoria_de_trabajo'] = data_set_mejorado['categoria_de_trabajo'].apply(lambda x: 'Sin categoria' if str(x) == 'nan' else x)

    # Missings en trabajo
    data_set_mejorado['trabajo'] = data_set_mejorado['trabajo'].apply(lambda x: 'No responde' if str(x) == 'nan' else x)
    # Eliminación de el atributo eduación alcanzada
    del data_set_mejorado['educacion_alcanzada']

    # Juntamos casado y casada en una misma categoría.
    data_set_mejorado['rol_familiar_registrado'] = data_set_mejorado['rol_familiar_registrado'].apply(lambda x : 'casado' if x == 'casada' else x)

    return data_set_mejorado
