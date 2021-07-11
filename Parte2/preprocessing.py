# +
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -

# Funcion de preprocessing basada en el analisis realizado en la parte 1.
def preprocessing_base_parte_1(X_train, X_test):
    X_train_preproc = X_train[['ganancia_perdida_declarada_bolsa_argentina', 'edad', 'rol_familiar_registrado', 'anios_estudiados']]
    X_train_preproc = pd.get_dummies(X_train_preproc, drop_first=True)

    X_test_preproc = X_test[['ganancia_perdida_declarada_bolsa_argentina', 'edad', 'rol_familiar_registrado', 'anios_estudiados']]
    X_test_preproc = pd.get_dummies(X_test_preproc, drop_first=True)

    return X_train_preproc, X_test_preproc


def standard_preprocessing_base_parte_1(X_train, X_test):
    X_train, X_test = preprocessing_base_parte_1(X_train, X_test)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    return X_train, X_test


# +
def _log_scale(x):
    return np.sign(x) * (np.log(abs(x)) + 1) if (x < -1 or x > 1) else x

# Se asume que a X_train y a X_test ya han sido dummificadas.  
def preprocessing_significantes(X_train, X_test, variance):
    pca_test = PCA()
    scaler = StandardScaler()

    X_train['ganancia_perdida_declarada_bolsa_argentina'] = X_train['ganancia_perdida_declarada_bolsa_argentina'].apply(_log_scale)
    X_test['ganancia_perdida_declarada_bolsa_argentina'] = X_test['ganancia_perdida_declarada_bolsa_argentina'].apply(_log_scale)
   
    X_train_preproc = scaler.fit_transform(X_train)
    X_test_preproc = scaler.transform(X_test)
    pca_test.fit(X_train_preproc)
    
    variances = np.where((np.cumsum(pca_test.explained_variance_ratio_) > variance)==True)
    pca = PCA(variances[0][0])

    X_train_preproc = pd.DataFrame(pca.fit_transform(X_train_preproc))
    X_test_preproc = pd.DataFrame(pca.transform(X_test_preproc))
    
    return X_train_preproc, X_test_preproc


# -

# Se asume que a X_train y a X_test ya han sido dummificadas.
def preprocessing_equilibrado(X_train, X_test, y_train, y_test):
    X_1 = X_train[y_train==1].copy()
    X_0 = X_train[y_train==0].copy()

    indexs = X_0.sample(len(X_1)).index
    X_train_preproc = pd.concat([X_1, X_0.loc[indexs]], ignore_index=True, sort=False)
    y_train_preproc = pd.concat([y_train[y_train==1],y_train.loc[indexs]])

    
    return X_train_preproc, X_test, y_train_preproc, y_test

def preprocessing_mejores_por_arbol(X_train, X_test):
    eleccion = ['estado_marital_matrimonio_civil', 'horas_trabajo_registradas',
                'trabajo_profesional_especializado', 'trabajo_directivo_gerente',
                'anios_estudiados', 'ganancia_perdida_declarada_bolsa_argentina',
                'edad', 'rol_familiar_registrado_casado']
    return X_train[eleccion].copy(), X_test[eleccion].copy()


def standard_preprocessing_mejores_por_arbol(X_train, X_test):
    X_train, X_test = preprocessing_mejores_por_arbol(X_train, X_test)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    return X_train, X_test
