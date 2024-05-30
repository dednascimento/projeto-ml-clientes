import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title='Classificação de Clientes',
    layout='wide'
)

# FUNÇÃO DE MODELO
@st.cache_data
def carregar_data_modelo():
    base = pd.read_csv('dados_clientes.csv', sep=',')
    encoder = OrdinalEncoder()
    base = base.drop('ID', axis=1)

    # CONVERTENDO AS COUNAS PARA INFORMAÇÕES CATEGÓRICAS
    for col in base.columns.drop('Categoria'):
        base[col] = base[col].astype('category')

    X_encoded = encoder.fit_transform(base.drop('Categoria', axis=1))
    y = base['Categoria'].astype("category").cat.codes

    # SEPARANDO DADOS DE TREINO E TESTE
    X_train, X_test, y_train, y_teste = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    # CRIANDO MODELO NB
    modelo = CategoricalNB()
    modelo.fit(X_train, y_train)

    # ARMAZENANDO A ACURÁCIA DO MODELO COM OS DADOS DE PREVISÃO
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_teste, y_pred)

    return encoder, modelo, acuracia, base

encoder, modelo, acuracia, base = carregar_data_modelo()

st.title('Classificação de Clientes')
st.write(f'Acurácia do modelo: {acuracia:.2f}')

input_features = [
    st.selectbox("Insira uma idade:", sorted(base['Idade'].unique())),
    st.selectbox("Insira sua Renda Anual (R$):", sorted(base['Renda Anual (R$)'].unique())),
    st.selectbox("Insira sua Renda Mensal (R$):", sorted(base['Gastos Mensais (R$)'].unique())),
    st.selectbox("Insira seu tempo conosco: (Meses):", sorted(base['Tempo como Cliente (meses)'].unique()))
]

# PROCESSAMENTO DE NOVAS INFROMAÇÕES
if st.button('Processar'):
    # CRIANDO DATAFRAME PARA PROCESSAMENTO DO MODELO
    input_df = pd.DataFrame([input_features], columns=base.columns.drop('Categoria'))

    # APLICANDO MODELO
    input_encoder = encoder.transform(input_df)
    predict_encoded = modelo.predict(input_encoder)

    # CONVERTENDO DE CATEGORIA NÚMERO Á NOME
    previsao = base['Categoria'].astype("category").cat.categories[predict_encoded][0]
    st.header(f'A classificação prevista é {previsao}.')


st.title('Base de Cliente Classificados')
st.dataframe(base, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.bar(base['Categoria'], base['Renda Anual (R$)'])
    st.pyplot(fig)
