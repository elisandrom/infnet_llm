# <b>IA generativa para linguagem (Large Language Model) [24E2_3]</b> 

## Projeto da Disciplina
Desenvolver uma aplicação utilizando Streamlit, LLM e LangChain.

## Objetivo
Neste projeto, foi desenvolvido um assistente gastronômico com o objetivo de auxiliar pessoas que desejam preparar pratos na cozinha, mas não conhecem o processo. O assistente fornece orientações passo a passo, receitas detalhadas e dicas culinárias, facilitando o preparo de refeições variadas. Além disso, o assistente pode sugerir receitas com base nos ingredientes informados e adaptar instruções para diferentes níveis de habilidade culinária, tornando a experiência culinária acessível e agradável para todos.


----
## Como Executar:
- Criar uma Key do serviço API Gemini pelo link: https://aistudio.google.com/app/apikey
- Adicionar a sua chave gerada na variável `agent` no arquivo `.\Questao7\src\app.py`
    <br>Trecho: `agent = Agent(google_api_key="sua-api-key-aqui")`
- Executar no terminal o seguinte comando: `streamlit run '.\Questao7\src\app.py'`