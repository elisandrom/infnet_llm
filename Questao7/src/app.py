import streamlit as st
from agent import Agent

agent = Agent(google_api_key="sua-api-key-aqui")

st.set_page_config(layout="wide")
st.title('IA na Cozinha')

st.write("Descubra como preparar os seus pratos preferidos na cozinha!")

col1, col2 = st.columns(2)

with col1:
    request = st.text_area("Qual receita você gostaria de preparar?")
    button = st.button("Enviar")
    box = st.container()

if button and request:
    with box:
        box = st.empty()
        box.header("Receita")
        box.write("Aguarde um momento...")
    preparation = agent.get_preparation(request)
    try:
        box.write(preparation['agent_response'])
    except KeyError:
        box.write("Desculpe, não consegui encontrar a receita para você.")