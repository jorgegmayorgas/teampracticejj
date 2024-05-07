import ast
from datetime import datetime
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from scipy.stats import ttest_ind,mannwhitneyu,ttest_rel,ttest_1samp,pearsonr
from scipy.stats import chi2_contingency,f_oneway
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix,ConfusionMatrixDisplay,r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,OrdinalEncoder


def internal_anova_columns (df:pd.DataFrame,target_col:str,lista_num_columnas:list=[],pvalue:float=0.05):
    """
    Uso interno, devuelve una lista con las columnas del dataframe cuyo ANOVA con la columna designada por "target_col" 
    supere el test de hipótesis con significación mayor o igual a 1-pvalue

    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas.
    `target_col` (str): Variable target tipo str.
    `pvalue` (float): Variable float con valor por defecto 0.05

    Retorna:
    col_nums: List
    """
    selected_columns=[]
    for columna in lista_num_columnas:
        unique_values=df[columna].unique()
        for valor in unique_values:
            t_statistic,p_value = f_oneway(df[target_col][df[columna]==valor],df[target_col])
            if p_value<pvalue:#corregido
                if columna not in selected_columns:
                    selected_columns.append(columna)
    return selected_columns

def internal_sns_pairplot(num_pair_plots:int,columns_per_plot:int,columns_for_pairplot:list,df:pd.DataFrame,target_col:str):
        
        for i in range(num_pair_plots):
            start_idx = i * columns_per_plot
            end_idx = (i + 1) * columns_per_plot
            current_columns = columns_for_pairplot[start_idx:end_idx + 1]  # Include the 'target' column
            # Create a pair plot with Seaborn for the current group of columns
            sns.set_theme(style="ticks")
            pair_plot = sns.pairplot(df, hue=target_col, vars=current_columns,palette='viridis')
            # Adjust layout and show the plot
            plt.tight_layout();
            plt.show();
        return plt

def internal_onehotencoder(df:pd.DataFrame,features_cat:list):
    """
    Turns categorical columns in binary columns

    Arguments:

        `df` (pd.DataFrame): Dataframe de Pandas.
        `features_cat` (list): List of categorical columns.

    Returns:
        df: pd.DataFrame
    """
    #Turns categorical columns in binary columns
    onehot = OneHotEncoder(sparse_output=False, drop='first') 
    data = onehot.fit_transform(df[features_cat])
    new_features = onehot.get_feature_names_out()
    df[new_features] = data
    df.drop(columns= features_cat, axis = 1, inplace = True)
    return df

def internal_create_pairplot(df:pd.DataFrame,target_col:str):
    cols = df.columns.drop(target_col)  # Exclude target_col from pairplot

    # Limit the number of columns to a maximum of 5
    cols = cols[:min(len(cols), 5)]  # Select at most 5 columns for the pairplot
    sns.set_theme(style="ticks")
    # Create the pairplot with hue based on target_col
    sns.pairplot(df, vars=cols, hue=target_col, palette="viridis")
    plt.tight_layout();
    #plt.suptitle(f"Pairplot for group: {group_data[target_col].unique()[0]}")  # Add group title
    plt.show();
    return plt

def internal_ordinalencoder(df:pd.DataFrame,target_col:str):
        for column in df.columns:
                unique_values=df[column].unique()
                encoder = OrdinalEncoder(categories=[unique_values])
                # Fit the encoder to the data and transform it
                encoded_data = encoder.fit_transform(df[[column]])
                df[column] = encoded_data.flatten()
        return df



def eval_model(features, target:str, problem_type, metrics, model):

    """
        Esta función debe recibir un target, unas predicciones para ese target, un argumento que determine si el problema es de regresión o clasificación y una lista de métricas:
        * Si el argumento dice que el problema es de regresión, la lista de métricas debe admitir las siguientes etiquetas RMSE, MAE, MAPE, GRAPH.
        * Si el argumento dice que el problema es de clasificación, la lista de métrica debe admitir, ACCURACY, PRECISION, RECALL, CLASS_REPORT, MATRIX, MATRIX_RECALL, MATRIX_PRED, PRECISION_X, RECALL_X. En el caso de las _X, X debe ser una etiqueta de alguna de las clases admitidas en el target.

        Funcionamiento:
        * Para cada etiqueta en la lista de métricas:
            - RMSE, debe printar por pantalla y devolver el RMSE de la predicción contra el target.
            - MAE, debe pintar por pantalla y devolver el MAE de la predicción contra el target. 
            - MAPE, debe pintar por pantalla y devolver el MAPE de la predcción contra el target. Si el MAPE no se pudiera calcular la función debe avisar lanzando un error con un mensaje aclaratorio
            - GRAPH, la función debe pintar una gráfica comparativa (scatter plot) del target con la predicción
            - ACCURACY, pintará el accuracy del modelo contra target y lo retornará.
            - PRECISION, pintará la precision media contra target y la retornará.
            - RECALL, pintará la recall media contra target y la retornará.
            - CLASS_REPORT, mostrará el classification report por pantalla.
            - MATRIX, mostrará la matriz de confusión con los valores absolutos por casilla.
            - MATRIX_RECALL, mostrará la matriz de confusión con los valores normalizados según el recall de cada fila (si usas ConfussionMatrixDisplay esto se consigue con normalize = "true")
            - MATRIX_PRED, mostrará la matriz de confusión con los valores normalizados según las predicciones por columna (si usas ConfussionMatrixDisplay esto se consigue con normalize = "pred")
            - PRECISION_X, mostrará la precisión para la clase etiquetada con el valor que sustituya a X (ej. PRECISION_0, mostrará la precisión de la clase 0)
            - RECALL_X, mostrará el recall para la clase etiquetada co nel valor que sustituya a X (ej. RECALL_red, mostrará el recall de la clase etiquetada como "red")

    Argumentos:

    `features` (list): Lista de features.
    `target` (str): Variable target tipo str.
    `problem_type` (str): Tipo de problema ['regression', 'classification']
    `metrics` (list): Lista de métricas
    `model` (ML model): Modelo de ML

    Retorna:
    Tupla: Tupla con métricas de regresión o clasificacion
    """

    # Comprobación del tipo de problema
    if problem_type not in ['regression', 'classification']:
        raise ValueError("El argumento 'problem_type' debe ser 'regression' o 'classification'.")

    # Comprobación de las métricas
    valid_regression_metrics = ['RMSE', 'MAE', 'MAPE', 'GRAPH']
    valid_classification_metrics = ['ACCURACY', 'PRECISION', 'RECALL', 'CLASS_REPORT', 'MATRIX', 'MATRIX_RECALL', 'MATRIX_PRED']

    for metric in metrics:
        if problem_type == 'regression' and metric not in valid_regression_metrics:
            raise ValueError(f"La métrica '{metric}' no es válida para un problema de regresión.")
        elif problem_type == 'classification' and metric not in valid_classification_metrics:
            raise ValueError(f"La métrica '{metric}' no es válida para un problema de clasificación.")

    # Obtener predicciones reales del modelo
    predictions = model.predict(features)

    # Métricas de regresión
    regression_metrics = ()
    if 'RMSE' in metrics:
        rmse = np.sqrt(mean_squared_error(target, predictions))
        print(f'RMSE: {rmse:.4f}')
        regression_metrics += (rmse,)

    if 'MAE' in metrics:
        mae = mean_absolute_error(target, predictions)
        print(f'MAE: {mae:.4f}')
        regression_metrics += (mae,)

    if 'MAPE' in metrics:
        try:
            mape = np.mean(np.abs((target - predictions) / target)) * 100
            print(f'MAPE: {mape:.4f}%')
            regression_metrics += (mape,)
        except ZeroDivisionError:
            raise ValueError("No se puede calcular MAPE cuando hay valores de target iguales a cero.")

    if 'GRAPH' in metrics:
        plt.scatter(target, predictions)
        plt.xlabel('Target')
        plt.ylabel('Predictions')
        plt.title('Comparativa entre Target y Predicciones')
        plt.show()

    # Métricas de clasificación
    classification_metrics = ()
    if 'ACCURACY' in metrics:
        accuracy = accuracy_score(target, predictions.round())
        print(f'Accuracy: {accuracy:.4f}')
        classification_metrics += (accuracy,)

    if 'PRECISION' in metrics:
        precision = precision_score(target, predictions.round(), average='macro')
        print(f'Precision: {precision:.4f}')
        classification_metrics += (precision,)

    if 'RECALL' in metrics:
        recall = recall_score(target, predictions.round(), average='macro')
        print(f'Recall: {recall:.4f}')
        classification_metrics += (recall,)

    if 'CLASS_REPORT' in metrics:
        print('Classification Report:')
        print(classification_report(target, predictions.round()))

    if 'MATRIX' in metrics:
        print('Confusion Matrix (Absolute Values):')
        print(confusion_matrix(target, predictions.round()))

    if 'MATRIX_RECALL' in metrics:
        disp = ConfusionMatrixDisplay(confusion_matrix(target, predictions.round(), normalize='true'))
        disp.plot(cmap='Blues', values_format='.2f', xticks_rotation='vertical')
        plt.title('Confusion Matrix (Normalized by Row - Recall)')
        plt.show()

    if 'MATRIX_PRED' in metrics:
        disp = ConfusionMatrixDisplay(confusion_matrix(target, predictions.round(), normalize='pred'))
        disp.plot(cmap='Blues', values_format='.2f', xticks_rotation='vertical')
        plt.title('Confusion Matrix (Normalized by Column - Predictions)')
        plt.show()

    # Métricas específicas de clasificación
    for metric in metrics:
        if metric == 'GRAPH':
            print("La métrica 'GRAPH' no es válida para un problema de clasificación.")

        elif 'PRECISION_' in metric:
            class_label = metric.split('_')[1]
            precision_class = precision_score(target, predictions.round(), labels=[class_label], average=None)[0]
            print(f'Precision {class_label}: {precision_class:.4f}')

        elif 'RECALL_' in metric:
            class_label = metric.split('_')[1]
            recall_class = recall_score(target, predictions.round(), labels=[class_label], average=None)[0]
            print(f'Recall {class_label}: {recall_class:.4f}')

    if problem_type == 'regression':
        return regression_metrics
    else:
        return classification_metrics
    
    
def get_features_num_classification (df:pd.DataFrame,target_col:str,pvalue:float=0.05):
    """
    La función devuelve una lista con las columnas numéricas del dataframe cuyo ANOVA con la columna designada por "target_col" 
    supere el test de hipótesis con significación mayor o igual a 1-pvalue

    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas.
    `target_col` (str): Variable target tipo str.
    `pvalue` (float): Variable float con valor por defecto 0.05.

    Retorna:
    col_nums: List
    """
    #Validaciones
    
    if target_col=="":
        raise ValueError("El parámetro target_col no puede estar vacío")
    if target_col not in df.columns:
        raise ValueError("El parámetro target_col debe estar en el DataFrame")
    #Búsqueda de columnas numéricas
    num_values=['float32','int32','float64','int64','int8', 'int16', 'float16','uint8', 'uint16', 'uint32', 'uint64']
    lista_num_columnas=[]
    for columna in df.columns:
        if df[columna].dtype in num_values:
            lista_num_columnas.append(columna)
    #ANOVA
    selected_columns=[]
    selected_columns= internal_anova_columns(df,target_col,lista_num_columnas,pvalue)
    '''
    for columna in lista_num_columnas:
        unique_values=df[columna].unique()
        for valor in unique_values:
            t_statistic,p_value = f_oneway(df[target_col][df[columna]==valor],df[target_col])
            if p_value<pvalue:#corregido
                if columna not in selected_columns:
                    selected_columns.append(columna)
    '''
    return selected_columns

def plot_features_num_classification (df:pd.DataFrame,target_col:str="",columns:list=[],pvalue:float=0.5):
    """
    La función pinta una pairplot del DataFrame considerando la columna designada por "target_col" y aquellas incluidas en "column" que cumplan el test de ANOVA 
    para el nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores. 
    Se espera que las columnas sean numéricas. El pairplot utiliza como argumento de hue el valor de target_col
    Si target_Col es superior a 5, se usan diferentes pairplot diferentes, se pinta un pairplot por cada 5 valores de target posibles.
    Si la lista de columnas a pintar es grande se pinten varios pairplot con un máximo de cinco columnas en cada pairplot,
    siendo siempre una de ellas la indicada por "target_col"

    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas.
    `target_col` (str): Variable target tipo str.
    `columns` (list): Variable con la lista de columnas de tipo list.
    `pvalue` (float): Variable float con valor por defecto 0.5.    
    
    Retorna:
    sns.pairplot: Pairplot
    """
    #ANOVA
    selected_columns= internal_anova_columns(df,target_col,columns,pvalue)
    columns_for_pairplot = df[selected_columns].columns
    if  len(columns_for_pairplot) <5:
        columns_per_plot =  len(columns_for_pairplot) 
    else:
        columns_per_plot = 5
    
    # Calculate the number of pair plots needed
    num_pair_plots = len(columns_for_pairplot) // columns_per_plot
    # Create pair plots for each group of 5 columns
    plt=internal_sns_pairplot(num_pair_plots,columns_per_plot,columns_for_pairplot,df,target_col)
    '''
    for i in range(num_pair_plots):
        start_idx = i * columns_per_plot
        end_idx = (i + 1) * columns_per_plot
        current_columns = columns_for_pairplot[start_idx:end_idx + 1]  # Include the 'target' column
        # Create a pair plot with Seaborn for the current group of columns
        sns.set_theme(style="ticks")
        pair_plot = sns.pairplot(df[current_columns], hue='target', palette='viridis')
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
    '''
    return plt

def internal_loop_create_pairplot(df:pd.DataFrame,categorical_cols:list,target_col:str):
    # Iterate through groups and create pairplots
        for column in df[categorical_cols].columns:
            #if df[column].dtype in ['object','category']:
                unique_values=df[column].unique()
                #print(f"unique_values:{unique_values}")
                encoder = OrdinalEncoder(categories=[unique_values])
                # Fit the encoder to the data and transform it
                encoded_data = encoder.fit_transform(df[[column]])
                df[column] = encoded_data.flatten()
    
        return internal_create_pairplot(df,target_col)

def plot_features_cat_classification(df:pd.DataFrame, target_col:str="", columns:list=[], mi_threshold:float=0.0, normalize:bool=False):
    """
    Esta función recibe un dataframe, una argumento "target_col" con valor por defecto "", una lista de strings ("columns") cuyo valor por defecto es la lista vacía, un argumento ("mi_threshold") con valor 0.0 por defecto, y un argumento "normalize" a False.

    Si la lista no está vacía:
    * La función seleccionará de esta lista los valores que correspondan a columnas o features categóricas del dataframe cuyo valor de mutual information respecto de target_col supere el umbral puesto en "mi_threshold" (con las mismas considereciones respecto a "normalize" que se comentan en la descripción de la función "get_features_cat_classification").
    * Para los valores seleccionados, pintará la distribución de etiquetas de cada valor respecto a los valores de la columna "target_col".

    Si la lista está vacía:
    * Entonces la función igualará "columns" a las variables categóricas del dataframe y se comportará como se describe en la sección "Si la lista no está vacía"

    De igual manera que en la función descrita anteriormente deberá hacer un check de los valores de entrada y comportarse como se describe en el último párrafo de la función `get_features_cat_classification`.
    """
    if target_col and df[target_col].dtype not in ['object', 'category']:
        print("Error: 'target_col' debe ser una variable categórica del DataFrame.")
        return None
    if not 0.0 <= mi_threshold <= 1.0 and normalize:
        print("Error: 'mi_threshold' debe estar entre 0 y 1 cuando 'normalize' es True.")
        return None
    
    if not columns:
        categorical_cols = get_features_cat_classification(df, target_col, mi_threshold, normalize)
    else:
        categorical_cols = columns
    print(categorical_cols)
    #corregir con get_features_cat_classification
    plt=None
    #selected_columns=get_features_cat_classification(df, target_col, mi_threshold, normalize)
    if len(categorical_cols)>0:
        plt=internal_loop_create_pairplot(df,categorical_cols,target_col)
        

    '''
    columns_for_pairplot = df[categorical_cols].columns
    print(columns_for_pairplot)
    columns_per_plot = 5
    # Calculate the number of pair plots needed
    num_pair_plots = len(columns_for_pairplot) // columns_per_plot

    plt=internal_sns_pairplot(num_pair_plots,columns_per_plot,columns,df,target_col)
    '''
    return plt

#Javier
def get_features_cat_classification(df:pd.DataFrame, target_col:str, mi_threshold:float=0.0, normalize:bool=False):
    '''
    Esta función recibe como argumentos un dataframe, el nombre de una de las columnas del mismo (argumento 'target_col'), que debería ser el target de un hipotético 
    modelo de clasificación, es decir debe ser una variable categórica o numérica discreta pero con baja cardinalidad, un argumento "normalize" con valor False por defecto, 
    una variable float "mi_threshold" cuyo valor por defecto será 0.
    * En caso de que "normalize" sea False:
        La función debe devolver una lista con las columnas categóricas del dataframe cuyo valor de mutual information con 'target_col' iguale o supere 
        el valor de "mi_threshold".
    * En caso de que "normalize" sea True:
        La función debe devolver una lista con las columnas categóricas del dataframe cuyo valor normalizado de mutual information con 'target_col' iguale o supere 
        el valor de "mi_threshold". 
        El valor normalizado de mutual information se considera el obtenido de dividir el valor de mutual information tal cual ofrece sklearn o la fórmula de cálculo 
        por la suma de todos los valores de mutual information de las features categóricas del dataframe.
    En este caso, la función debe comprobar que "mi_threshold" es un valor float entre 0 y 1, y arrojar un error si no lo es.
    La función debe hacer todas las comprobaciones necesarias para no dar error como consecuecia de los valores de entrada.
    Es decir hará un check de los valores asignados a los argumentos de entrada y si estos no son adecuados debe retornar None y printar 
    por pantalla la razón de este comportamiento. Ojo entre las comprobaciones debe estar que "target_col" hace referencia a una variable numérica continua del dataframe.
    
    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas.
    `target_col` (str): Variable target tipo str.
    `mi_threshold` (float): Variable float con valor por defecto 0.0.
    `normalize` (bool): Variable bool con valor por defecto False.
    `relative` (bool): Variable bool con valor por defecto False.

    Retorna:
    
    selected_columns: List

    '''

    # Comprobación de argumentos de entrada

    if target_col not in df.columns:
        raise TypeError("'target_col' debe ser una variable categórica o numérica discreta del DataFrame.")
        return None
    
    if not 0.0 <= mi_threshold <= 1.0 and normalize:
        raise TypeError("'mi_threshold' debe estar entre 0 y 1 cuando 'normalize' es True.")
        return None
    
    selected_columns=[]
    #tmp_cat_cols=[]
    #tmp_cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols=[]
    for cat_col in df.columns:
        if df[cat_col].nunique()<=10:
            cat_cols.append(cat_col)
    #print(f"df.columns:{df.columns}")
    #print(f"cat_cols:{cat_cols}")
    if len(cat_cols)>0:
        #df=internal_onehotencoder(df,cat_cols)
        print(f"2cat_cols:{cat_cols}")
        df[cat_cols]=internal_ordinalencoder(df[cat_cols].copy(),target_col)
        #print(f"2df.columns:{df.columns}")
        if normalize==False:

            for columna in cat_cols:
                if columna!=target_col:
                        mi_score_categorical = mutual_info_classif(df[[columna]], df[target_col])
                        #print(f"mi_score_categorical_{columna}:{mi_score_categorical}")
                        if mi_score_categorical>=mi_threshold:
                            if columna not in selected_columns:
                                selected_columns.append(columna)
        
        else:
            list_mi_score_categorical=[]
            #print(f"3cat_cols:{cat_cols}")
            #print(f"3df.columns:{df.columns}")
            for columna in cat_cols:
                if columna!=target_col:
                        mi_score_categorical = mutual_info_classif(df[[columna]], df[target_col])
                        list_mi_score_categorical.append(mi_score_categorical)
            #print(f"list_mi_score_categorical:{list_mi_score_categorical}")
            for columna in cat_cols:
                if columna!=target_col:
                    mi_score_categorical = mutual_info_classif(df[[columna]], df[target_col])
                    mi_score_categorical_normalized = mi_score_categorical/sum(list_mi_score_categorical)
                    if mi_score_categorical_normalized>=mi_threshold:
                            if columna not in selected_columns:
                                selected_columns.append(columna)
            
        
    return selected_columns
    


#version inicial toolbox_ML.py
def describe_df(df:pd.DataFrame):
    """
    Función que devuelve un Dataframe  de Pandas con:

        * El tipo de datos por columna
    
        * El porcentaje de nulos
        
        * La cantidad de valores únicos
        
        * La cardinalidad de la columna en porcentaje

    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas a describir.
    

    Retorna:
    pandas.DataFrame: Dataframe
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"El parámetro debe ser un Dataframe de Pandas")
    tipo_de_dato = df.dtypes
    valores_nulos = (df.isna().mean() * 100).round(2)
    valores_unicos = df.nunique()
    cardinalidad = ((df.nunique()/df.shape[0])*100).round(2)

    descripcion = pd.DataFrame({
        'Tipo de dato': tipo_de_dato,
        'Valores nulos (%)': valores_nulos,
        'Valores unicos': valores_unicos,
        'Cardinalidad (%)': cardinalidad
    })

    return descripcion.T
    

def get_features_num_regression(df:pd.DataFrame, target_col:str, umbral_corr:float, pvalue=None):
    """
    Función que devuelve una lista con las columnas numéricas 
    del dataframe cuya correlación con la columna designada 
    por "target_col" sea superior en valor absoluto al valor dado 
    por "umbral_corr". 
    
    Además si la variable "pvalue" es distinta de None, sólo devolvera 
    las columnas numéricas cuya correlación supere el valor indicado y 
    además supere el test de hipótesis con significación 
    mayor o igual a 1-pvalue

    Argumentos:
    
    `df` (pandas.DataFrame): Variable que contiene dataframe de Pandas.
    
    `target_col` (list): Nombre de la columna target de un modelo de regresión
    
    `umbral_corr` (float): Umbral de correlación, entre 0 y 1.
    
    `pvalue` (float): Descripción de param1.

    Retorna:
    
    list: Lista de Python
    """
    # Verificamos el tipo en la variable df
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"El primer parámetro debe ser un Dataframe de Pandas")
    # Verificamos el tipo en la variable target_col
    if not isinstance(target_col, str):
        raise TypeError(f"El parámetro target_col debe ser un string")
    # Verificamos el tipo en la variable umbral_corr
    if not isinstance(umbral_corr, float):
        raise TypeError(f"El parámetro umbral_corr debe ser un float")

                        
    # Verificamos si existe algún error al llamar a la columna 'target_col'
    if target_col not in df.columns:
        raise TypeError(f"Error: La columna '{target_col}' no está bien indicada, no se puede asignar como 'target_col'.")


    # Verificamos si 'target_col' es una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        raise TypeError(f"Error: La columna '{target_col}' no es una variable numérica continua.")


    # Verificamos si 'umbral_corr' está en el rango [0, 1]
    if not (0 <= umbral_corr <= 1):
        raise TypeError("Error: Se ha indicado un 'umbral_corr' incorrecto, debe estar entre el rango [0, 1].")

    
    # Verificamos si 'pvalue' es None, float o int y además un valor válido
    if pvalue is not None and (not isinstance(pvalue, (float, int)) or pvalue <= 0 or pvalue >= 1):
        raise TypeError("Error: Si 'pvalue' no es 'None', debe tener un valor entre (0, 1).")


    # Obtenemos las columnas numéricas del dataframe
    cols_numericas = df.select_dtypes(include=[np.number]).columns

    # Calculamos la correlación y p-value para cada columna numérica con 'target_col'
    correlaciones = []
    for col in cols_numericas:
        corr, p_value = pearsonr(df[col], df[target_col])
        correlaciones.append((col, corr, p_value))

    # Filtramos las columnas basadas en 'umbral_corr' y 'pvalue'
    features_seleccionadas = []
    for col, corr, p_value in correlaciones:
        #if abs(corr) > umbral_corr and (pvalue is None or p_value < 1 - pvalue):#corregir
        if abs(corr) > umbral_corr and (pvalue is None or p_value < pvalue):#corregido
            features_seleccionadas.append((col, corr, p_value))

    # Devolvemos la lista de características seleccionadas junto con sus correlaciones y valores p
    return features_seleccionadas


def plot_features_num_regression(df:pd.DataFrame,target_col="", columns=list(""), umbral_corr=0.0,pvalue=None):
    """
    Función que devuelve una lista con las columnas numéricas 
    del dataframe cuya correlación con la columna designada 
    por "target_col" sea superior en valor absoluto al valor dado 
    por "umbral_corr". 
    
    Además si la variable "pvalue" es distinta de None, sólo devolvera 
    las columnas numéricas cuya correlación supere el valor indicado y 
    además supere el test de hipótesis con significación 
    mayor o igual a 1-pvalue

    Argumentos:
    
    `df` (pandas.DataFrame): Variable que contiene dataframe de Pandas.
    
    `target_col` (string): Nombre de la columna target de un modelo de regresión
    
    `columns` (list): Lista con los nombres de las columnas
    
    `umbral_corr` (float): Umbral de correlación, entre 0 y 1. Por defecto valor 0
    
    `pvalue` (int): Valor de significación estadística

    Retorna:

    sns.pairplot: Pairplot
    """
    
    # Verificamos si existe algún error al llamar a la columna 'target_col'
# Verificamos si 'target_col' es una variable numérica continua
    if target_col and not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es una variable numérica continua.")
        return None
    
    # Verificamos si 'umbral_corr' está en el rango [0, 1]
    if not (0 <= umbral_corr <= 1):
        print("Error: Se ha indicado un 'umbral_corr' incorrecto, debe estar entre el rango [0, 1].")
        return None
    
    # Verificamos si 'pvalue' es un valor válido
    if pvalue is not None and (not isinstance(pvalue, (float, int)) or pvalue <= 0 or pvalue >= 1):
        print("Error: Si 'pvalue' no es 'None', debe tener un valor entre (0, 1).")
        return None
    
    # Si 'target_col' está presente, excluimos esa columna
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col:
        columns.remove(target_col)
    
    # Filtramos las columnas basadas en 'umbral_corr' y 'pvalue'
    selected_columns = []
    for col in columns:
        if target_col:
            corr, p_value = pearsonr(df[col], df[target_col])
            if abs(corr) > umbral_corr and (pvalue is None or p_value < pvalue):#corregido
                selected_columns.append(col)
    
    # Si no hay columnas seleccionadas, mostramos un mensaje y devolvemos None
    if not selected_columns:
        print("No hay columnas que cumplan con los criterios de selección.")
        return None
    
    # Pintamos los pairplots
    sns.pairplot(df[selected_columns + [target_col]])
    
    return selected_columns


def get_features_cat_regression(df:pd.DataFrame,target_col:str="", pvalue=0.05):
    """
    Función que devuelve una lista con las columnas categóricas del 
    dataframe cuyo test de relación con la columna designada por 
    'target_col' supere en confianza estadística el test de relación 
    que sea necesario 
    
    Argumentos:
    
    `df` (pandas.DataFrame): Variable que contiene dataframe de Pandas.
    
    `target_col` (string): Nombre de la columna target de un modelo de regresión
    
    `pvalue` (float): Valor de significación estadística. Por defecto valor 0.05

    Retorna:

    list: columnas categóricas del dataframe cuyo test de relación con la columna designada por 'target_col' supere en confianza estadística el test de relación que sea necesario hacer
    """
    
    lista_de_categoricas=list()
    # Verificamos si existe algún error al llamar a la columna 'target_col'
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no está bien indicada, no se puede asignar como 'target_col'.")
        return None
    # Verificamos si 'target_col' es una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es una variable numérica continua.")
        return None
    # Verificamos si 'pvalue' es un valor válido
    if pvalue is not None and (not isinstance(pvalue, (float, int)) or pvalue <= 0 or pvalue >= 1):
        print("Error: Si 'pvalue' no es 'None', debe tener un valor entre (0, 1).")
        return None
    #realizar test de relación de confianza
    #print("Test de correlación")
    df_categoricas=df.drop(target_col,axis=1)
    
    for columna in df_categoricas.columns:
        if not np.issubdtype(df[columna].dtype, np.number):
            if not np.issubdtype(df[columna].dtype, np.datetime64):
                lista_de_categoricas.append(columna)
            #print("lista_de_categoricas:",lista_de_categoricas)

    
    selected_columns = []
    
    for columna in lista_de_categoricas:#averiguo los valores únicos
        unique_values=df[columna].unique()
        parametros="" # para formar la llamada dinámicamente a t-test y chi cuadrado
        for valor in unique_values:
            #parametros=parametros + f'df["{target_col}"][df["{columna}"]=="{valor}"]'
            #if valor != unique_values[len(unique_values)-1]:
            #    parametros=parametros + ","
        
        #print("parametros:",parametros)

        # Create a dictionary to store variable values
        #variables = {}
        # Execute the dynamic code and capture variables in the dictionary
        #exec(f"t_statistic,p_value = f_oneway({parametros})", globals(), variables) #este si
        #exec(instructions,{},variables) #este si
            t_statistic,p_value = f_oneway(df[target_col][df[columna]==valor],df[target_col])
            if p_value<pvalue:#corregido
                if columna not in selected_columns:
                    selected_columns.append(columna)
            else:
                #sentencia=f"chi2_stat, p_value, dof, expected = chi2_contingency(pd.crosstab(df['{columna}'], 
                # df['{target_col}']))"#no es lo correcto u-mann whitney
                #print("sentencia:",sentencia)
                #exec(sentencia, globals(), variables)
                stat, p_value = mannwhitneyu(df[target_col][df[columna]==valor],df[target_col])
                if p_value<pvalue:#corregido
                    if columna not in selected_columns:
                        selected_columns.append(columna)
    
    if not selected_columns:
        print("No hay columnas que cumplan con los criterios de selección.")
        return None      
    else:
        return selected_columns


def plot_features_cat_regression(df:pd.DataFrame,target_col:str="", columns:list=[],pvalue=0.05,with_individual_plot:bool=False):
    """
    Función que pintará los histogramas agrupados de la variable "target_col"
    para cada uno de los valores de las variables categóricas incluidas en 
    columns que cumplan que su test de relación con "target_col" es 
    significativo para el nivel 1-pvalue de significación estadística. 
    La función devolverá los valores de "columns" que cumplan con las 
    condiciones anteriores. 
    
    Argumentos:
    
    df (pandas.DataFrame): Variable que contiene dataframe de Pandas.
    
    target_col (string): Nombre de la columna target de un modelo de regresión
    
    columns(list): Lista con los nombres de las columnas
    
    pvalue (float): Por defecto valor 0.05

    with_indivual_plot(boolean): Por defector valor False

    Retorna:

    sns.pairplot: Pairplot
    """
    # Verificamos si existe algún error al llamar a la columna 'target_col'
    
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no está bien indicada, no se puede asignar como 'target_col'.")
        return None

    # Verificamos si 'target_col' es una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es una variable numérica continua.")
        return None

    # Verificamos si 'pvalue' es un valor válido
    if pvalue is not None and (not isinstance(pvalue, (float, int)) or pvalue <= 0 or pvalue >= 1):
        print("Error: Si 'pvalue' no es 'None', debe tener un valor entre (0, 1).")
        return None
    #print("Entro")
    columnas_cat = []
    for col in columns:
        tabla = pd.crosstab(df[col], df[target_col])
        chi2, p_value_col, _, _ = chi2_contingency(tabla)
        #print("chi2_contingency(tabla)[1] < pvalue",chi2_contingency(tabla)[1])
        #if chi2_contingency(tabla)[1] > pvalue:
        if p_value_col < pvalue:#corregido
            columnas_cat.append(col)
            if with_individual_plot:
                for col in columnas_cat:    
                    plt.figure(figsize=(10, 8))
                    sns.barplot(x=target_col, y=col, data=df)
                    plt.xlabel('Target')
                    plt.ylabel("Col")
                    plt.title("Relación entre target y 'col'")

                    plt.tight_layout()
                    plt.show()
                    
    return columnas_cat             
def datetime_naming_convention():
    """
    Returns datetime on a string with format YYYYMMDDD_HHMMSS

    Arguments:

        None

    Returns:
        formatted_datime: str
    """
# Get the current date and time
    current_datetime = datetime.now()
# Format the date and time as a string in the desired format
    formatted_datetime = current_datetime.strftime('%Y%m%d_%H%M%S')

    return formatted_datetime

def arima_metrics(y_test,model_name:str,column:str,value_of_column:str,preds):
    """
    Metrics for ARIMA, SARIMA and AARIMA models

    Arguments:

        `df` (pd.DataFrame):            Pandas DataFrame
        `model`(ARIMA/SARIMA/AARIMA) :  ML model ARIMA/SARIMA/AARIMA
        `column` (str):                 If present add aditional column to identify results
        `value_of_column`:              Value of the additional column

    Returns:
        `df`:                           DataFrame with results of metrics

    """
    results_dict={}
    results_list=[]
    results_dict['model']={model_name}
    y_test[np.isinf(y_test)] = 0
    preds[np.isinf(preds)] = 0
    try:
        results_dict['r_square_score']=round(r2_score(y_test, preds), 4)
    except (ZeroDivisionError,ValueError,OverflowError):
        results_dict['r_square_score']=0
    try:
        results_dict['mae_score']=round(mean_absolute_error(y_test, preds), 4)
    except (ZeroDivisionError,ValueError,OverflowError):
        results_dict['mae_score']=0
    try:
        results_dict['mse_score']=round(mean_squared_error(y_test, preds), 4)
    except (ZeroDivisionError,ValueError,OverflowError):
        results_dict['mse_score']=0
    try:        
        results_dict['rmse_score']=round(np.sqrt(mean_squared_error(y_test, preds)), 4)
    except (ZeroDivisionError,ValueError,OverflowError):
        results_dict['rmse_score']=0
    try:
        results_dict['mape_score']=round(mean_absolute_percentage_error(y_test, preds), 4)
    except (ZeroDivisionError,ValueError,OverflowError):
        results_dict['mape_score']=0
    if column !='' and value_of_column !='':
        results_dict[column]=value_of_column
    
    results_list.append(results_dict)
    df_metrics = pd.DataFrame(results_list)
    return df_metrics

def model_metrics(model_metrics:dict,model_name:str,column:str,value_of_column:str):
    """
    Metrics for ML models

    Arguments:

        `model_metrics` (Dict):         Dictionary with values
        `model_name` :                  ML model name
        `column` (str):                 If present add aditional column to identify results
        `value_of_column`:              Value of the additional column

    Returns:
        `df`:                           DataFrame with results of metrics

    """
    results_dict={}
    results_list=[]
    results_dict['model']=model_name
    results_dict['r_square_score']=model_metrics['r_square_score']
    results_dict['mae_score']=model_metrics['mae_score']
    results_dict['mse_score']=model_metrics['mse_score']
    results_dict['rmse_score']=model_metrics['rmse_score']
    results_dict['mape_score']=model_metrics['mape_score']
    
    if column !='' and value_of_column !='':
        results_dict[column]=value_of_column
    
    results_list.append(results_dict)
    df_metrics = pd.DataFrame(results_list)
    return df_metrics

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_test_vs_preds(y_test,preds,key:str="",log_or_sqrt:str=""):
        """
        Metrics for ML models

        Arguments:

            `y_test` (Any):                 Column target of a dataframe
            `preds`(Any) :                  Predictions of a ML model
            `key` (str):                    Model name
            `log_or_sqrt`:                  String with "log" or "sqrt" values

        Returns:
            `plt`:                          Plt object

        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, preds, alpha=0.5)
        plt.title(f'Predictions vs Actual Values {key} {log_or_sqrt}')
        plt.xlabel(f'Actual Values')
        plt.ylabel(f'Predicted Values')
        # Plot a line for perfect predictions
        plt.plot(y_test, y_test, color='red')
        plt.show()
def function_models_log_sqrt(df:pd.DataFrame,target,drop_columns:list,bool_log:True,bool_sqrt:False):

    if (bool_log and bool_sqrt) or (bool_log==False and bool_sqrt==False):
        raise ValueError("bool_log and bool_sqrt cannot be True/False at the same time")
        return None
    drop_columns=drop_columns+target

    
    df_features = df.drop(drop_columns,axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_features,#X
                                                        df[target].squeeze(),#Y
                                                        test_size=0.2,
                                                        random_state=42)

    X_train_scaled,X_test_scaled=dafu.scaler_of_x_train_and_x_test(X_train,X_test)

    rfr_model = RandomForestRegressor(random_state=42)
    lr_model = LinearRegression()
    dtr_model = DecisionTreeRegressor(random_state=42)
    gbr_model = GradientBoostingRegressor(random_state=42)
    knr_model=KNeighborsRegressor()
    xgb_model=XGBRegressor(random_state=42)
    lgb_model=LGBMRegressor(random_state=42,verbose=-100)
    cbr_model=CatBoostRegressor(random_state=42,verbose=False)
    #svr_model = SVR()
    elastic_model = ElasticNet()
    ridge_model = Ridge()
    lasso_model = Lasso()
    mlp_model = MLPRegressor()
    tfk_model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.12),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.04)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.02)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model

    tfk_model.compile(optimizer='adam', loss='mean_squared_error')


    # Train the model
    models={}
    models={'RandomForestRegressor':rfr_model,
                'LinearRegression':lr_model,
                'DecisionTreeRegressor':dtr_model,
                'GradientBoostingRegressor':gbr_model,
                'KNeighborsRegressor':knr_model,
                'XGBRegressor':xgb_model,
                'LGBMRegressor':lgb_model,
                'CatBoostRegressor':cbr_model,
                'Ridge':ridge_model,
                'Lasso':lasso_model,
                'MLP':mlp_model,
                'ElasticNet' : elastic_model,
                'Keras':tfk_model
                }

    list_df_metrics=[]
    for key,model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        #if bool_log:
        #    original_scale_predictions = np.exp(preds)
        #    original_y_test=np.exp(y_test)
        #if bool_sqrt:
        #    original_scale_predictions = np.square(preds)
        #    original_y_test=np.square(y_test)
    # Metrics
        metrics_dict={}
        value_r2_score=r2_score(y_test, preds)
        value_mae_score=mean_absolute_error(y_test, preds)
        value_mse_score=mean_squared_error(y_test, preds)
        value_rmse_score=np.sqrt(mean_squared_error(y_test, preds))
        value_mape_score=tb.mean_absolute_percentage_error(y_test, preds)
        metrics_dict['r_square_score']=value_r2_score
        metrics_dict['mae_score']=value_mae_score
        metrics_dict['mse_score']=value_mse_score
        metrics_dict['rmse_score']=value_rmse_score
        metrics_dict['mape_score']=value_mape_score
        print(f"{key} R^2 score:", round(value_r2_score, 4))
        print(f"{key} MAE score:", round(value_mae_score, 4))
        print(f"{key} MSE score:", round(value_mse_score, 4))
        print(f"{key} RMSE score:", round(value_rmse_score, 4))
        print(f"{key} MAPE score:", round(value_mape_score, 4))
        if bool_log:
            list_df_metrics.append(tb.model_metrics(metrics_dict,key,"transformation","log"))
            log_or_sqrt="log"
            
        if bool_sqrt:
            list_df_metrics.append(tb.model_metrics(metrics_dict,key,"transformation","sqrt"))
            
            log_or_sqrt="sqrt"
        filename = f'./metrics/{tb.datetime_naming_convention()}_{key}_{log_or_sqrt}'
        ###

        # Create a scatter plot
        tb.plot_test_vs_preds(y_test,preds,key,log_or_sqrt)

        ###
            #list_df_metrics.append(tb.arima_metrics(original_y_test,key,"transformation","square",original_scale_predictions))
        #if key in ['LinearRegression','Ridge', 'Lasso', 'ElasticNet','KNeighborsRegressor']:
        #    importances=model.coef_
        #else:
        #    importances =model.feature_importances_
        #feature_importances = pd.DataFrame(importances, index=df_features.columns.to_list(), columns=["importance"]).sort_values("importance", ascending=False)        #print(f"{key}",importances)

        if key != 'Keras':  # Keras models uses another format
            filename = f'{filename}.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
        else:
            filename = f'{filename}.h5'
            model.save(filename)  # Saves the Keras model
        print(f"filename:{filename}")
    df_metrics=pd.concat(list_df_metrics, ignore_index=True)
    df_metrics.to_csv(f'./metrics/{tb.datetime_naming_convention()}_metrics_{log_or_sqrt}.csv')
    dict_data_function={}
    
    try:
        dict_data_function={
        'models':models,
        'X_train':X_train, 
        'X_test':X_test, 
        'y_train':y_train, 
        'y_test':y_test,
        'X_train_scaled':X_train_scaled,
        'X_test_scaled':X_test_scaled,
    }
    except ImportError:
        dict_data_function={
        'models':0,
        'X_train':0, 
        'X_test':0, 
        'y_train':0, 
        'y_test':0,
        'X_train_scaled':0,
        'X_test_scaled':0
        }
    print("Success")
    return dict_data_function

def calculate_correlation_groups(df:pd.DataFrame, target, feature_columns, group_size=10):
    #feature_columns = [col for col in df.columns if col != 'target']
    correlation_results = {}

    # Split the feature columns into groups of 10
    for i in range(0, len(feature_columns), group_size):
        group = feature_columns[i:i+group_size]
        group.append(target)  # Add the target variable to each group
        subgroup = df[group]
        correlation_matrix = subgroup.corr().sort_values(ascending=False)
        correlation_results[f'Group {i//group_size + 1}'] = correlation_matrix

    return correlation_results