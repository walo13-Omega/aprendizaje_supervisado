from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Crear la app Flask
app = Flask(__name__)

# Cargar el modelo Pipeline entrenado
try:
    with open("pipeline.pkl", "rb") as archivo_modelo:
        modelo = pickle.load(archivo_modelo)
    print("✅ Pipeline cargado correctamente.")
except FileNotFoundError:
    print("🚫 No se encontró el archivo pipeline.pkl. Asegúrate de haberlo generado con 5_pipeline.ipynb.")
    modelo = None


@app.route("/")
def home():
    return jsonify({
        "mensaje": "Bienvenido al API del modelo Titanic 🛳️",
        "endpoint_disponible": "/predecir",
        "metodo": "POST"
    })


@app.route("/predecir", methods=["POST"])
def predecir():
    if modelo is None:
        return jsonify({"error": "El modelo no está cargado correctamente."}), 500

    # Obtener el JSON de la solicitud
    data = request.get_json()

    # Validar que los campos requeridos estén presentes
    columnas_esperadas = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    if not all(col in data for col in columnas_esperadas):
        return jsonify({
            "error": f"Faltan columnas. Se esperaban: {columnas_esperadas}"
        }), 400

    # Crear DataFrame con los datos recibidos
    input_data = pd.DataFrame([data])

    # Realizar predicción
    try:
        prediccion = modelo.predict(input_data)
        resultado = int(prediccion[0])
    except Exception as e:
        return jsonify({"error": f"Ocurrió un problema al predecir: {str(e)}"}), 500

    # Respuesta JSON
    return jsonify({
        "input": data,
        "prediccion": resultado,
        "mensaje": "1 = Sobrevive, 0 = No sobrevive"
    })


if __name__ == "__main__":
    app.run(debug=True)
    