
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import requests

st.set_page_config(page_title="Análise Jurídica com IA", layout="wide")

st.title("📊 Análise de Dados Jurídicos + IA com Streamlit")

menu = st.sidebar.radio("Navegação", ["🔍 Análise Exploratória", "🤖 Previsão com IA"])

file = st.sidebar.file_uploader("📁 Envie seu arquivo CSV", type=["csv"])

if file:
    df = pd.read_csv(file, parse_dates=['data_abertura', 'data_ultima_movimentacao', 'data_encerramento'])
    df['ano_abertura'] = df['data_abertura'].dt.year
    df['mes_abertura'] = df['data_abertura'].dt.month

    if menu == "🔍 Análise Exploratória":
        st.header("📊 Análise Exploratória de Dados")

        with st.expander("🧹 Visão Geral do Dataset"):
            st.write(df.head())
            st.write(df.info())
            st.write("Valores nulos por coluna:")
            st.write(df.isnull().sum())

        with st.expander("📑 Distribuição dos Tipos de Processo"):
            fig, ax = plt.subplots()
            sns.countplot(y='tipo_processo', data=df, order=df['tipo_processo'].value_counts().index, ax=ax)
            st.pyplot(fig)

        with st.expander("⚖️ Áreas Jurídicas"):
            fig, ax = plt.subplots()
            sns.countplot(y='area_juridica', data=df, order=df['area_juridica'].value_counts().index, ax=ax)
            st.pyplot(fig)

        with st.expander("🗺️ Mapa de Processos por UF"):
            try:
                geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
                uf_geojson = requests.get(geojson_url).json()

                uf_data = df['uf_processo'].value_counts().reset_index()
                uf_data.columns = ['uf_processo', 'qtd_processos']

                fig = px.choropleth(
                    uf_data,
                    geojson=uf_geojson,
                    locations='uf_processo',
                    featureidkey='properties.sigla',
                    color='qtd_processos',
                    color_continuous_scale="Reds",
                    title='Distribuição de Processos por Estado (UF)',
                    scope="south america"
                )
                fig.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Erro ao gerar o mapa: {e}")

        with st.expander("⏱️ Tempo de Tramitação dos Processos Encerrados"):
            encerrados = df[df['status_processo'] == 'Encerrado']
            fig, ax = plt.subplots()
            sns.histplot(encerrados['tempo_tramitacao_dias'], kde=True, bins=30, ax=ax)
            st.pyplot(fig)

        with st.expander("💰 Valor da Causa por Área Jurídica"):
            fig, ax = plt.subplots()
            sns.boxplot(data=df[df['valor_causa'] < 500000], x='valor_causa', y='area_juridica', ax=ax)
            st.pyplot(fig)

    elif menu == "🤖 Previsão com IA":
        st.header("🤖 Modelo de Previsão do Status do Processo")

        with st.spinner("Treinando modelo..."):
            model_df = df[['tipo_processo', 'area_juridica', 'valor_causa', 'honorarios_advogado', 'provisao_risco', 'status_processo']].dropna()

            le = LabelEncoder()
            model_df['tipo_processo'] = le.fit_transform(model_df['tipo_processo'])
            model_df['area_juridica'] = le.fit_transform(model_df['area_juridica'])
            model_df['status_processo'] = le.fit_transform(model_df['status_processo'])

            X = model_df.drop('status_processo', axis=1)
            y = model_df['status_processo']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📊 Resultados do Modelo")
            st.text("Matriz de Confusão")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("Relatório de Classificação")
            st.text(classification_report(y_test, y_pred))
else:
    st.warning("Por favor, envie um arquivo `.csv` para começar.")
