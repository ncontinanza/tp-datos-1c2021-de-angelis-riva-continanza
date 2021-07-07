import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Funcion de preprocessing basada en el analisis realizado en la parte 1.
def preprocessing_base_parte_1(df):
    df_preproc = df[['ganancia_perdida_declarada_bolsa_argentina', 'edad', 'rol_familiar_registrado', 'anios_estudiados', 'tiene_alto_valor_adquisitivo']]
    df_preproc = pd.get_dummies(df_preproc, drop_first=True)
    return df_preproc


def _log_scale(x):
    return np.sign(x) * (np.log(abs(x)) + 1) if (x < -1 or x > 1) else x

def preprocessing_significantes(df):
    scaler = MinMaxScaler()
    pca = PCA(30)
    
    pa = df['tiene_alto_valor_adquisitivo']
    del df['tiene_alto_valor_adquisitivo']
    
    df = pd.get_dummies(df, drop_first=True)
    df['ganancia_perdida_declarada_bolsa_argentina'] = df['ganancia_perdida_declarada_bolsa_argentina'].apply(_log_scale)

    df = scaler.fit_transform(df)
    return (pd.DataFrame(pca.fit_transform(df)), pa)
