from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    file_path = './data/BD_SINDES_clean.csv'  # Cambia esta ruta si es necesario
    data = pd.read_csv(file_path)

    # Selecciona los primeros 100 registros
    data_limited = data.head(30)

    # Convierte a lista de diccionarios para pasar a la plantilla
    results = data_limited.to_dict(orient='records')
    
    return render_template('index.html',plots=[
        'static/plots/perdida_entrenamiento_validacion.png',
        'static/plots/valores_reales_vs_predichos.png',
        'static/plots/grafico_residuos.png'
    ],results = results)

if __name__ == '__main__':
    app.run(debug=True)
