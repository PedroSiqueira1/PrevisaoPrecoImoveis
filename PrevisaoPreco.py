
# Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# Importando dados

dados = pd.read_csv("ImóvelDados\conjunto_de_treinamento.csv")
dados_teste = pd.read_csv("ImóvelDados\conjunto_de_teste.csv")
exemplo = pd.read_csv("ImóvelDados\exemplo_arquivo_respostas.csv")


# Definindo índices

dados.set_index('Id', inplace=True)
dados_teste.set_index('Id', inplace=True)

# Análise atributos

for coluna in dados.columns:
    print(coluna)
    print(dados[coluna].value_counts())
    print(dados.groupby(coluna).preco.mean())
    
for coluna in dados.columns:
    print(pd.isnull(dados[coluna]).sum())
    
# Limpando dados    

dados = dados.drop(['bairro', # One Hot encoding muito grande
                    'diferenciais'],axis=1) # Informação já binarizada em outras colunas


dados_teste = dados_teste.drop(['bairro', # One Hot encoding muito grande
                                'diferenciais'],axis=1) # Informação já binarizada em outras colunas

# One Hot Encoding

dados = pd.get_dummies(dados,columns=['tipo',
                                      'tipo_vendedor'])

dados_teste = pd.get_dummies(dados_teste,columns=['tipo',
                                      'tipo_vendedor'])


dados_teste['tipo_Quitinete'] = 0 # Adicionando coluna para deixar as tabelas de dados e dados_teste iguais

# Selecionando atributos

atributos_selecionados = ['quartos', 
                          'suites', 
                          'vagas', 
                          'area_util', 
                          'area_extra',
                           'churrasqueira', 
                           'estacionamento', 
                           'piscina', 
                           'playground', 
                           'quadra',
                           's_festas', 
                           's_jogos', 
                           's_ginastica', 
                           'sauna', 
                           'vista_mar',
                           'tipo_Apartamento', 
                           'tipo_Casa', 
                           'tipo_Loft', 
                           'tipo_Quitinete',
                           'tipo_vendedor_Imobiliaria', 
                           'tipo_vendedor_Pessoa Fisica',
                           'preco']

dados = dados[atributos_selecionados]
dados_teste = dados_teste[atributos_selecionados[0:-1]]

# Separando alvo e atributos

X = dados.iloc[:,dados.columns != 'preco'].values
y = dados.iloc[:,dados.columns == 'preco'].values.ravel()


X_train, X_test, y_train,y_test = train_test_split(X,y, train_size=3300, random_state = 42)

# Ajustar escala

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
dados_teste = scaler.transform(dados_teste)

# Função para ver resultados

def resultados(grid):
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    results = results[['params','mean_test_score','std_test_score']].head()
    
    return results


# RandomForest

# Best 	params {'criterion': 'mae', 'max_depth': 4, 'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 500}


parametersRF = {
    "max_depth":[4],
    "max_features": ['auto'],
    "n_estimators": [500],
    "min_samples_leaf":[5],
    "criterion":['mae']
    }

gridRF = GridSearchCV(RandomForestRegressor(), parametersRF, cv=2, verbose=2, n_jobs =-1)
gridRF.fit(X, y)

resultadosRF = resultados(gridRF)
bestRF = gridRF.best_estimator_
bestRF.fit(X,y)
respostaRF = bestRF.predict(dados_teste)

# Resposta para csv

resposta_final = respostaRF

exemplo['preco'] = respostaRF
exemplo.to_csv('resposta1.csv', index= False)