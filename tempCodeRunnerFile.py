# app.py
from http.server import BaseHTTPRequestHandler, HTTPServer

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello, World!')

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run()

import pandas 
import pandas as pd
import numpy as np
import seaborn  as sns
import matplotlib.pyplot as plt 
import os
from graphviz import Source

from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report



df =  pd.read_excel('datateste1.xlsx')

df.drop("BDNF (pg/mL)", axis = 1, inplace=True)
df.drop("Irisin (ng/mL)", axis = 1, inplace=True)
df.drop("FABP3 (pg/mL)", axis = 1, inplace=True)
df.drop("FABP4 (pg/mL)", axis = 1, inplace=True)
df.drop("Oxytocin (pg/mL)", axis = 1, inplace=True)
df.drop("Leptin (pg/mL)", axis = 1, inplace=True)
df.drop("IL-8 (pg/mL)", axis = 1, inplace=True)
df.drop("IL-6 (pg/mL)", axis = 1, inplace=True)
df.drop("IP10 (pg/mL)", axis = 1, inplace=True)
df.drop("MCP1 (pg/mL)", axis = 1, inplace=True)
df.drop("MIP1b (pg/mL)", axis = 1, inplace=True)
df.drop("RANTES (pg/mL)", axis = 1, inplace=True)
df.drop("VEGF (pg/mL)", axis = 1, inplace=True)
df.drop("Pan-ApoE (ug/mL)", axis = 1, inplace=True)
df.drop("ApoE4 (ug/mL)", axis = 1, inplace=True)
df.drop("ApoE4/ApoE (Pan-ApoE)", axis = 1, inplace=True)
df.drop("ApoE4 pheno (type)", axis = 1, inplace=True)
df.drop("Ab42/Ab40", axis = 1, inplace=True)
df.drop("Noradrenaline (ng/mL)", axis = 1, inplace=True)
df.drop("L-Dopa", axis = 1, inplace=True)
df.drop("Dopamine", axis = 1, inplace=True)
df.drop("Dopac", axis = 1, inplace=True)
df.drop("5-HIAA", axis = 1, inplace=True)
df.drop("HVA", axis = 1, inplace=True)
df.drop("Serotonine", axis = 1, inplace=True)
df.drop("HVA/DA", axis = 1, inplace=True)
df.drop("Dopac+HVA/DA", axis = 1, inplace=True)
df.drop("5-HIAA/5-HT", axis = 1, inplace=True)
df.drop("Glutamate (μM)", axis = 1, inplace=True)
df.drop("Glutamine", axis = 1, inplace=True)
df.drop("Taurine", axis = 1, inplace=True)
df.drop("Arginine", axis = 1, inplace=True)
df.drop("GABA", axis = 1, inplace=True)
df.drop("Glutamate/GABA", axis = 1, inplace=True)
df.drop("Glutamine/ Glutamate ", axis = 1, inplace=True)
df.drop("A7/A5", axis = 1, inplace=True)
df.drop("MMSE", axis = 1, inplace=True)
df.drop("Ab/tau", axis = 1, inplace=True)
df.drop("Glutamine/ GABA", axis = 1, inplace=True)
df.drop("Lipoxin A4  (pg/mL)", axis = 1, inplace=True)
df.drop("Cys-LT (pg/mL)", axis = 1, inplace=True)
df.drop("LXA4/cys-LT", axis = 1, inplace=True)
df.drop("GABA/ Glutamate", axis = 1, inplace=True)
df.drop("Total protein (mg/mL)", axis = 1, inplace=True)
df.drop("Subjects", axis = 1, inplace=True)
df.drop(labels=25, axis=0, inplace=True)
df.drop(labels=26, axis=0, inplace=True)
df.drop(labels=27, axis=0, inplace=True)
df.drop(labels=28, axis=0, inplace=True)
df.drop(labels=29, axis=0, inplace=True)
df.drop(labels=30, axis=0, inplace=True)
df.drop(labels=31, axis=0, inplace=True)
df.drop(labels=32, axis=0, inplace=True)
df.drop(labels=33, axis=0, inplace=True)
df.drop(labels=34, axis=0, inplace=True)
df.drop(labels=35, axis=0, inplace=True)
df.drop(labels=36, axis=0, inplace=True)
df.drop(labels=37, axis=0, inplace=True)
df.drop(labels=38, axis=0, inplace=True)
df.drop(labels=53, axis=0, inplace=True)
df.drop(labels=54, axis=0, inplace=True)
df.drop(labels=55, axis=0, inplace=True)
df.drop(labels=56, axis=0, inplace=True)
df.drop(labels=57, axis=0, inplace=True)
df.drop(labels=58, axis=0, inplace=True)
df.drop(labels=59, axis=0, inplace=True)
df.drop(labels=60, axis=0, inplace=True)
df.drop(labels=61, axis=0, inplace=True)

#exclui intervalos de linhas - exemplo: df.drop(df.index[2:4], inplace=True) - para não ter esse trabalho todo. 

from sklearn.preprocessing import OrdinalEncoder
from category_encoders import OrdinalEncoder
maplist = [{'col': 'DX', 'mapping': {'NDC': 1, 'AD': 0}} ]
oe = OrdinalEncoder(mapping=maplist)
data_df= oe.fit_transform(df)
print(data_df)
X = data_df.iloc[:,1:5].values
Previsor = data_df.iloc[:,5:6].values
# test_size (porcentagem dos dados que irão para teste) e o random_state (define se a divisão vai ser embaralhada toda vez que o programa for executado)
x_treinamento, x_teste, y_treinamento, y_teste =  train_test_split(X, Previsor, test_size=0.30, random_state=7)


#StandardScaler remove a média e dimensiona cada recurso/variável para a variação da unidade. Esta operação é realizada de forma independente .
#StandardScaler pode ser influenciado por outliers (se existirem no conjunto de dados), pois envolve a estimativa da média empírica e desvio padrão de cada característica.


#Os outliers são dados que se diferenciam drasticamente de todos os outros. Em outras palavras, um outlier é um valor que foge da normalidade e que pode (e provavelmente irá) causar anomalias nos resultados obtidos por meio de algoritmos e sistemas de análise.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_treinamento = sc.fit_transform(x_treinamento)
x_teste = sc.transform(x_teste)
print(x_treinamento)


#Entender melhor o random_state, testei com 10 deu uma acurácia de 0,75 e com 15 e 16 deu uma de 0,83. Como influencia? 

model = RandomForestClassifier()

model.fit(x_treinamento, y_treinamento.ravel())
preds = model.predict(x_teste)
#modelo.score - ela serve para sabermos se o nosso modelo será efetivo ao receber um dado que ele nunca viu na vida
print (model.score(x_treinamento, y_treinamento)) 
print(model.score(x_teste, y_teste))

arvore = model.estimators_[5]

from graphviz import Source
from sklearn.tree import export_graphviz

#x = pd.data_df(data_df[['Sex', 'Age (Years)', 'Ab40 (ng/mL)', 'Ab42 (ng/mL)', 'Total tau (ng/mL) ', 'DX']])
y = data_df.columns
print(y)


export_graphviz(arvore,
                out_file='tree.dot',
                max_depth=1,
                rounded=True,
                filled=True,
                impurity=False,
                feature_names = data_df.columns[0:4],
                proportion=True,
                class_names=["não", "sim"]
                )

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
#Max_depth=2 2 perguntas que a máquina irá fazer para tomar a decisão
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('tree.png')
imgplot = plt.imshow(img)
print(plt.show())

#filled : Leva um valor booleano. Se True, ele pinta o nó para indicar a classe majoritária para classificação, extremidade de valores para regressão ou pureza do nó para saída múltipla.

Algoritmo_floresta_aleatoria = RandomForestClassifier(n_estimators=400)
print(Algoritmo_floresta_aleatoria.fit(x_treinamento, y_treinamento.ravel()))
Previsoes = Algoritmo_floresta_aleatoria.predict(x_teste)
Matriz_Confusao = confusion_matrix(y_teste, Previsoes)
print(Matriz_Confusao)


ConfusionMatrixDisplay(Matriz_Confusao ).plot();

plt.show()

report = classification_report( y_teste, Previsoes)
print(report)


Previsão_adni = pd.read_excel('adniteste.xlsx')
print(Previsão_adni.head())

Prever = Previsão_adni.iloc[:,4:5].values
Previsão_adni['Previsão do Modelo'] = Algoritmo_floresta_aleatoria.predict(Prever)
print(Previsão_adni)

print(Previsão_adni ['Previsão do Modelo'].value_counts())
