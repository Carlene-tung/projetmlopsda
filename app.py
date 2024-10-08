from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd

app = Flask(__name__)

# Charger le modèle depuis le fichier de manière plus sécurisée
try:
    with open("logistic_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    app.logger.error(f"Erreur lors du chargement du modèle : {e}")

def validate_input(form):
    """ Valide et convertit les entrées du formulaire. """
    try:
        features = {
            'credit_lines_outstanding': int(form.get("credit_lines_outstanding", 0)),
            'loan_amt_outstanding': float(form.get("loan_amt_outstanding", 0.0)),
            'total_debt_outstanding': float(form.get("total_debt_outstanding", 0.0)),
            'income': float(form.get("income", 0.0)),
            'years_employed': int(form.get("years_employed", 0)),
            'fico_score': int(form.get("fico_score", 0))
        }
        return features
    except ValueError as e:
        app.logger.error(f"Erreur de saisie des données : {e}")
        raise

def model_pred(features):
    """ Effectue la prédiction en utilisant le modèle chargé. """
    if model:
        test_data = pd.DataFrame([features])
        prediction = model.predict(test_data)
        return int(prediction[0])
    return None

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            features = validate_input(request.form)
            prediction = model_pred(features)
            if prediction is None:
                return render_template("error.html", error="Modèle non disponible.")

            prediction_text = "Le client pourrait être en défaut de paiement." if prediction == 1 else "Le client ne présente pas de risque de défaut."
            return render_template("index.html", prediction_text=prediction_text)
        except ValueError:
            return render_template("error.html", error="Erreur de saisie des données.")
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
