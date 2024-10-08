import pytest
import pandas as pd
import pickle
from app import model_pred

# Fixture pour charger le modèle une seule fois pour tous les tests
@pytest.fixture(scope="module")
def loaded_model():
    model_path = "logistic_model.pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# Fonction test modifiée pour utiliser le modèle chargé via fixture
def test_predict(loaded_model):
    # Exemple de données à tester
    new_data = {
        'credit_lines_outstanding': 1,
        'loan_amt_outstanding': 3659.97,
        'total_debt_outstanding': 6785.83,
        'income': 62270.67,
        'years_employed': 5,
        'fico_score': 639,
    }
    
    # Conversion des données en DataFrame pour la prédiction
    test_data = pd.DataFrame([new_data])
    
    # Prédiction en utilisant le modèle chargé
    prediction = loaded_model.predict(test_data)
    
    # Conversion de la prédiction en entier et vérification
    assert int(prediction[0]) == 0, "La prédiction est incorrecte, elle devrait être 0"

# Fonction additionnelle pour tester un cas d'erreur
def test_predict_failure_case(loaded_model):
    # Données conçues pour tester un échec de prédiction attendu
    bad_data = {
        'credit_lines_outstanding': 10,
        'loan_amt_outstanding': 50000,
        'total_debt_outstanding': 80000,
        'income': 30000,
        'years_employed': 1,
        'fico_score': 300,
    }
    
    test_data = pd.DataFrame([bad_data])
    prediction = loaded_model.predict(test_data)
    assert int(prediction[0]) != 0, "Erreur attendue non détectée pour une prédiction qui devrait échouer"
