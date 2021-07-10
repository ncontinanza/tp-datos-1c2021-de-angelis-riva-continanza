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


# +
def _log_scale(x):
    return np.sign(x) * (np.log(abs(x)) + 1) if (x < -1 or x > 1) else x

def preprocessing_significantes(X_train, X_test, variance):
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    pca_test = PCA()
    scaler = StandardScaler()
    
    X_train['ganancia_perdida_declarada_bolsa_argentina'] = X_train['ganancia_perdida_declarada_bolsa_argentina'].apply(_log_scale)
    X_test['ganancia_perdida_declarada_bolsa_argentina'] = X_test['ganancia_perdida_declarada_bolsa_argentina'].apply(_log_scale)
    
    X_train_preproc = scaler.fit_transform(X_train)
    X_test_preproc = scaler.transform(X_test)
    
    pca_test.fit(X_train_preproc)
    
    n = (np.cumsum(pca.explained_variance_ratio_) > variance).index(True)
    pca = PCA(n)
    
    X_train_preproc = pd.DataFrame(pca.fit_transform(X_train_preproc))
    X_test_preproc = pd.DataFrame(pca.transform(X_test_preproc))
    
    return X_train_preproc, X_test_preproc


# -

def preprocessing_equilibrado(X_train, X_test, y_train, y_test):
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    X_1 = X_train[y_train==1].copy()
    X_0 = X_train[y_train==0].copy()

    indexs = X_0.sample(len(X_1)).index
    X_train_preproc = pd.concat([X_1, X_0.loc[indexs]], ignore_index=True, sort=False)
    y_train_preproc = pd.concat([y_train[y_train==1],y_train.loc[indexs]])

    
    return X_train_preproc, y_train_preproc, X_test, y_test


def preprocessing_4_mejores_variables_arbol(X_train, X_test):
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    eleccion = ['anios_estudiados', 'ganancia_perdida_declarada_bolsa_argentina', 'edad', 'rol_familiar_registrado_casado']
    return X_train[eleccion].copy(), X_test[eleccion].copy()


def preprocessing_mejor_separacion_tSNE(X_train, X_test):
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    eleccion = ['anios_estudiados', 'ganancia_perdida_declarada_bolsa_argentina', 'edad',
                'rol_familiar_registrado_casado', 'religion_budismo','trabajo_entretenimiento',
                'rol_familiar_registrado_otro'] 
    return X_train[eleccion].copy(), X_test[eleccion].copy()


