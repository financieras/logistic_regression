import math
import numpy as np
import pandas as pd
import json

###### FUNCTIONS FOR DESCRIPTIVE STATISTICS ######

def ft_count(data):
    """Calculate the number of non-null observations."""
    return sum(1 for x in data if x is not None and not math.isnan(x))

def ft_mean(data):
    """Calculate the arithmetic mean of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    return float('nan') if not clean_data else sum(clean_data) / len(clean_data)

def ft_std(data):
    """Calculate the standard deviation of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 2:
        return float('nan')
    mean = ft_mean(clean_data)
    return math.sqrt(sum((x - mean) ** 2 for x in clean_data) / (len(clean_data) - 1))

def ft_min(data):
    """Find the minimum value in the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    return float('nan') if not clean_data else min(clean_data)

def ft_max(data):
    """Find the maximum value in the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    return float('nan') if not clean_data else max(clean_data)

def ft_percentile(data, q):
    """Calculate the qth percentile of the data."""
    clean_data = sorted([x for x in data if x is not None and not math.isnan(x)])
    if not clean_data:
        return float('nan')
    if len(clean_data) == 1:
        return clean_data[0]
    position = (len(clean_data) - 1) * q
    floor, ceil = math.floor(position), math.ceil(position)
    if floor == ceil:
        return clean_data[int(position)]
    d0 = clean_data[floor] * (ceil - position)
    d1 = clean_data[ceil] * (position - floor)
    return d0 + d1

def ft_median(data):
    """Calculate the median (50th percentile) of the data."""
    return ft_percentile(data, 0.5)

def ft_iqr(data):
    """Calculate the Interquartile Range (IQR) of the data."""
    q75, q25 = ft_percentile(data, 0.75), ft_percentile(data, 0.25)
    return float('nan') if math.isnan(q75) or math.isnan(q25) else q75 - q25

def ft_skewness(data):
    """Calculate the skewness of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 3:
        return float('nan')
    mean, std = ft_mean(clean_data), ft_std(clean_data)
    if std == 0:
        return float('nan')
    m3 = sum((x - mean) ** 3 for x in clean_data) / len(clean_data)
    return m3 / (std ** 3)

def ft_kurtosis(data):
    """Calculate the kurtosis of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 4:
        return float('nan')
    mean, std = ft_mean(clean_data), ft_std(clean_data)
    if std == 0:
        return float('nan')
    m4 = sum((x - mean) ** 4 for x in clean_data) / len(clean_data)
    return (m4 / (std ** 4)) - 3

def ft_cv(data):
    """Calculate the Coefficient of Variation (CV) of the data."""
    mean, std = ft_mean(data), ft_std(data)
    return float('nan') if mean == 0 or math.isnan(mean) or math.isnan(std) else abs(std / mean)



###### FUNCTIONS FOR LOGISTIC REGRESSION ######

def softmax(z):
    """
    Calcula la función softmax para clasificación multinomial
    El cálculo se realiza para cada fila de la matriz de entrada

    Parámetros:
    z: matriz de forma (n_muestras, n_clases)

    Retorna:
    matriz de probabilidades de forma (n_muestras, n_clases)
    donde cada fila suma 1
    """
    # Restar el máximo de cada fila para estabilidad numérica
    # Esto evita desbordamiento en exp() con números grandes
    z_shifted = z - np.max(z, axis=1, keepdims=True)

    # Calculamos exp() de los valores desplazados
    exp_scores = np.exp(z_shifted)

    # Normalizamos dividiendo por la suma
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def compute_cost(X, y, W):
    """
    Calcula la función de pérdida logarítmica (cross-entropy) para clasificación multinomial

    Parámetros:
    X: matriz de características (incluyendo columna de 1's) de forma (n_muestras, n_características)
    y: matriz one-hot de etiquetas reales de forma (n_muestras, n_clases)
    W: matriz de pesos de forma (n_características, n_clases)

    Retorna:
    J: valor de la función de pérdida
    """
    m = X.shape[0]  # número de muestras

    # Calcular predicciones
    z = np.dot(X, W)  # (n_muestras, n_clases)
    h = softmax(z)    # (n_muestras, n_clases)

    # Calcular pérdida logarítmica
    epsilon = 1e-15  # para evitar log(0)

    # Multiplicación elemento a elemento de y real con log de predicciones
    # y sumamos sobre todas las clases (axis=1) y todas las muestras
    J = -(1/m) * np.sum(y * np.log(h + epsilon))

    return J



###### FUNCTIONS FOR PREDICT ######

def predict(X, W):
    """
    Realiza predicciones usando los pesos aprendidos
    Parámetros:
    X: matriz de características (incluyendo columna de 1's)
    W: matriz de pesos optimizada
    Retorna:
    predicciones: matriz de probabilidades para cada clase
    """
    z = np.dot(X, W)
    return softmax(z)


def prepare_test_data(df_test):
    """
    Prepara los datos de test para las predicciones
    """
    X_test = df_test[['Best Hand', 'Age', 'Herbology', 'Defense Against the Dark Arts',
                      'Potions', 'Charms', 'Flying']]
    X_test = np.c_[np.ones(len(X_test)), X_test]
    return np.array(X_test)

def make_house_predictions(X_test, W_optimal):
    """
    Realiza las predicciones y devuelve tanto las casas predichas como las probabilidades
    """
    # Hacer predicciones usando los pesos óptimos
    probabilities = predict(X_test, W_optimal)
    predicted_houses = np.argmax(probabilities, axis=1)
    
    # Convertir índices a nombres de casas
    house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    predictions = [house_names[idx] for idx in predicted_houses]
    
    return predictions, probabilities

def save_predictions(predictions, output_file):
    """
    Guarda las predicciones en un archivo CSV
    """
    predictions_df = pd.DataFrame({
        'Index': range(len(predictions)),
        'Hogwarts House': predictions
    })
    predictions_df.to_csv(output_file, index=False)
    return predictions_df

def compare_predictions(file_paths):
    """
    Compara las predicciones de diferentes archivos CSV
    """
    dfs = [pd.read_csv(file) for file in file_paths]
    
    if dfs[0].equals(dfs[1]) and dfs[1].equals(dfs[2]):
        print("\n✅ Los tres archivos CSV son idénticos.")
    else:
        print("\n❌ Los archivos tienen diferencias.")
        for i in range(1, len(dfs)):
            diffs = dfs[0].compare(dfs[i])
            if not diffs.empty:
                print(f"\n🔍 Diferencias entre {file_paths[0]} y {file_paths[i]}:")
                print(diffs)

def save_weights(W, output_file='../output/model_weights.json'):
    """
    Guarda los pesos del modelo en formato JSON
    
    Parámetros:
    W: matriz de pesos numpy del modelo
    output_file: ruta completa del archivo donde guardar los pesos. Por defecto es '../output/model_weights.json'
    """    
    # Convertir matriz de pesos numpy a lista
    weights = W.tolist()
    
    # Guardar en JSON
    with open(output_file, 'w') as f:
        json.dump(weights, f)