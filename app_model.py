import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import seaborn as sns
import subprocess
import urllib.request
from flask import Flask, jsonify, request,render_template, send_from_directory
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
####
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,MultiLabelBinarizer,OneHotEncoder,OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage.io import imread
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix,ConfusionMatrixDisplay,r2_score, mean_absolute_error, mean_squared_error,mean_absolute_percentage_error,r2_score
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.decomposition import PCA


#import scripts.bootcampviztools as bc
#import scripts.data_functions as dafu
#import scripts.toolbox_ML as tb


# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True
root_path = "/home/jorgegmayorgas/teampracticejj/"
root_path ="/home/jorge/teampracticejj/"
label_dict={'setosa':0,'versicolor':1,'virginica':2}
label_dict_reverse={0:'setosa',1:'versicolor',2:'virginica'}
# Enruta la landing page (endpoint /)
@app.route('/', methods=['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    
    #html='<!DOCTYPE html><head><title>BOOOOOO 2</title></head><body>Bienvenido a mi API del modelo advertising!!! <link rel="apple-touch-icon" sizes="180x180" href="icons/apple-touch-icon.png"> <link rel="icon" type="image/png" sizes="32x32" href="icons/favicon-32x32.png"> <link rel="icon" type="image/png" sizes="16x16" href="icons/favicon-16x16.png"> <link rel="manifest" href="icons/site.webmanifest"></body>'
    #return html
    #landing_path = root_path + "/landing/"
    #full_path= landing_path + "index.html"
    #with open(full_path, 'r', encoding='utf-8') as file:
    #    html_page = file.read()
    #return html_page
    return render_template('index.html')

# Enruta la funcion al endpoint /api/v1/predicta
@app.route('/api/v1/predict', methods=['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET

    model = pickle.load(open(root_path + '20240505_091754_random_forest.pkl','rb'))
    lsepal = request.args.get('lsepal')
    wsepal = request.args.get('wsepal')
    lpetal = request.args.get('lpetal')
    wpetal = request.args.get('wpetal')
    print(lsepal,wsepal,lpetal,wpetal)
    #print(type(tv))
    bln_error=False
    if lsepal is None:
        bln_error=True
    if wsepal is None:
        bln_error=True
    if lpetal is None:
        bln_error=True
    if wpetal is None:
        bln_error=True

    dict_get_values={'sepal_length_(cm)':[lsepal],
    'sepal_width_(cm)':[wsepal],
    'petal_length_(cm)':[lpetal],
    'petal_width_(cm)':[wpetal]}
    df_get_values=pd.DataFrame(dict_get_values)
    prediction = model.predict(df_get_values)
    result_json={'prediction': prediction[0]}
    #return jsonify({'predictions_label': label_dict_reverse[prediction[0]],'predictions': prediction[0]})
    return result_json
# Enruta la funcion al endpoint /api/v1/labelflowerclasses
@app.route('/api/v1/labelflowerclasses', methods=['GET'])
def labelflowerclasses(): # Ligado al endpoint '/api/v1/predict', con el método GET

    #return jsonify({'predictions_label': label_dict_reverse[prediction[0]],'predictions': prediction[0]})
    return jsonify(label_dict)
# Enruta la funcion al endpoint /api/v1/labelflowerclasses
@app.route('/api/v1/numericflowerclasses', methods=['GET'])
def numericflowerclasses(): # Ligado al endpoint '/api/v1/predict', con el método GET

    
    #return jsonify({'predictions_label': label_dict_reverse[prediction[0]],'predictions': prediction[0]})
    return jsonify(label_dict_reverse)
# Enruta la funcion al endpoint /api/v1/retrain
@app.route('/api/v1/retrain/', methods=['GET'])
def retrain(): # Rutarlo al endpoint '/api/v1/retrain/', metodo GET
    #if os.path.exists(root_path + "data/Advertising_new.csv"):
    #    data = pd.read_csv(root_path + 'data/Advertising_new.csv')
    #
    #        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                            #data['sales'],
                                                            #test_size = 0.20,
                                                            #random_state=42)

            #model = Lasso(alpha=6000)
            #model.fit(X_train, y_train)
            #rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            #mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
            #model.fit(data.drop(columns=['sales']), data['sales'])
            #pickle.dump(model, open(root_path + 'ad_model.pkl', 'wb'))

            #return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
        #else:
            #return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"
            return None
@app.route('/webhook_2024', methods=['POST'])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = '/home/jorgegmayorgas/teampracticejj'
    servidor_web = '/var/www/jorgegmayorgas_pythonanywhere_com_wsgi.py' 

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if 'repository' in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']
            
            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'El directorio del repositorio no existe'}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull', clone_url], check=True)
                subprocess.run(['touch', servidor_web], check=True) # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f'Se realizó un git pull en el repositorio {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error al realizar git pull en el repositorio {repo_name}'}), 500
        else:
            return jsonify({'message': 'No se encontró información sobre el repositorio en la carga útil (payload)'}), 400
    else:
        return jsonify({'message': 'La solicitud no contiene datos JSON'}), 400
if __name__=="__main__":
    #app.run()
    app.run(host='0.0.0.0', port=5000)