import pandas as pd

# Funcion de preprocessing basada en el analisis realizado en la parte 1.
def preprocessing_base_parte_1(df):
    df_preproc = df[['ganancia_perdida_declarada_bolsa_argentina', 'edad', 'rol_familiar_registrado', 'anios_estudiados', 'tiene_alto_valor_adquisitivo']]
    df_preproc = pd.get_dummies(df_preproc, drop_first=True)
    return df_preproc

# Eliminamos las variables que consideramos que no nos aportan variabilidad
def preprocessing_significantes(df):
    df_preproc = df[['ganancia_perdida_declarada_bolsa_argentina', 'edad', 'rol_familiar_registrado', 'anios_estudiados', 'genero', 'tiene_alto_valor_adquisitivo']]
    df_preproc = pd.get_dummies(df_preproc, drop_first=True)
    return df_preproc

# Tenemos en cuenta la representatividad poblacional y la estimamos
def preprocessing_rep_poblacional(df):
    return
