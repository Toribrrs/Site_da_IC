from flask import Flask, request, jsonify, redirect, url_for, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

# Carregar e processar os dados
df = pd.read_excel('datateste1.xlsx')
columns_to_drop = [
    "BDNF (pg/mL)", "Irisin (ng/mL)", "FABP3 (pg/mL)", "FABP4 (pg/mL)", "Oxytocin (pg/mL)",
    "Leptin (pg/mL)", "IL-8 (pg/mL)", "IL-6 (pg/mL)", "IP10 (pg/mL)", "MCP1 (pg/mL)",
    "MIP1b (pg/mL)", "RANTES (pg/mL)", "VEGF (pg/mL)", "Pan-ApoE (ug/mL)", "ApoE4 (ug/mL)",
    "ApoE4/ApoE (Pan-ApoE)", "ApoE4 pheno (type)", "Ab42/Ab40", "Noradrenaline (ng/mL)",
    "L-Dopa", "Dopamine", "Dopac", "5-HIAA", "HVA", "Serotonine", "HVA/DA", "Dopac+HVA/DA",
    "5-HIAA/5-HT", "Glutamate (μM)", "Glutamine", "Taurine", "Arginine", "GABA",
    "Glutamate/GABA", "Glutamine/ Glutamate", "A7/A5", "MMSE", "Ab/tau", "Glutamine/ GABA",
    "Lipoxin A4  (pg/mL)", "Cys-LT (pg/mL)", "LXA4/cys-LT", "GABA/ Glutamate",
    "Total protein (mg/mL)", "Subjects"
]
df.drop(columns_to_drop, axis=1, inplace=True)
df.drop([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 53, 54, 55, 56, 57, 58, 59, 60, 61], inplace=True)

X = df.iloc[:, 1:5].values
Previsor = df.iloc[:, 5:6].values

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(X, Previsor, test_size=0.30, random_state=7)

sc = StandardScaler()
x_treinamento = sc.fit_transform(x_treinamento)
x_teste = sc.transform(x_teste)

model = RandomForestClassifier()
model.fit(x_treinamento, y_treinamento.ravel())

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    proteina1 = data['proteina1']
    proteina2 = data['proteina2']
    proteina3 = data['proteina3']
    values = sc.transform([[proteina1, proteina2, proteina3]])
    prediction = model.predict(values)[0]
    return jsonify({"prediction": prediction})

@app.route('/results.html')
def show_result():
    prediction = request.args.get('prediction')
    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
