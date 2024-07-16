import os
import warnings
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import Counter
from transformers import pipeline

##---------------------------------------------------
#Inicializações
##---------------------------------------------------
warnings.filterwarnings("ignore")
tqdm.pandas()
##---------------------------------------------------

##---------------------------------------------------
#Funções de Apoio
##---------------------------------------------------
def extract_entities_original(text):
    entities = ner_model(text) 
    return [entity for entity in entities if "ORG" in entity['entity_group']]


def extract_entities(text):
    entities = ner_model(text)  
    """
        Definições das entitys:

        B-ORG (Begin-Organization): Indica o início de uma entidade do tipo "Organização".
        I-ORG (Inside-Organization): Indica que o token faz parte de uma entidade do tipo "Organização", mas não é o primeiro token dessa entidade.
        L-ORG (Last-Organization): Indica que o token é o último da entidade do tipo "Organização".
        U-ORG (Unit-Organization): Indica que o token é uma entidade do tipo "Organização" composta por um único token.
    """
    org_entities = []
    current_org = ""
    current_start = None
    for entity in entities:
        if entity['entity'] in ['B-ORG', 'I-ORG', 'L-ORG', 'U-ORG']:
            if current_start is None:
                current_start = entity['start']
            current_org += text[entity['start']:entity['end']]
            if entity['entity'] == 'L-ORG' or entity['entity'] == 'U-ORG':
                # Ajustar início se necessário
                while current_start > 0 and text[current_start - 1] not in (" ", "\n", "\t"):
                    current_start -= 1
                org_entities.append(text[current_start:entity['end']])
                current_org = ""
                current_start = None
        else:
            if current_org:
                # Ajustar início se necessário
                while current_start > 0 and text[current_start - 1] not in (" ", "\n", "\t"):
                    current_start -= 1
                org_entities.append(text[current_start:entity['end']])
                current_org = ""
                current_start = None
    if current_org:
        while current_start > 0 and text[current_start - 1] not in (" ", "\n", "\t"):
            current_start -= 1
        org_entities.append(text[current_start:entity['end']])

    org_entities = [item.strip().replace("\"","").replace("\t","") for item in org_entities]

    return org_entities
    

##------------------------------------------------------------------


#Verifica se possui como rodar nos CUDA Cores ou se somente na CPU
env_GPU_CPU = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Possui Cuda disponível: " + str(torch.cuda.is_available()))
print("Modo de execução da pipeline: " + str(env_GPU_CPU))

#Faz a leitura do CSV
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data','articles.zip'))
#print(df.info())
#print(df.head())
#print(df['category'].unique())

#Arruma as definições das colunas importantes
df['date'] = pd.to_datetime(df['date'])

#Filtra as notícias
df_mercado = df[(df['category'] == 'mercado') & (df['date'].dt.year == 2015) & (df['date'].dt.month.isin([1, 2, 3]))]

#Limpa os indexes para começar do zero novamente
df_mercado = df_mercado.reset_index(drop=True)
print(df_mercado.head())

print(f"Total de Linhas no DataSet Original: {df.shape[0]}")
print(f"Total de Linhas no DataSet Filtrado - Mercado: {df_mercado.shape[0]}")

#Executa o pipeline no modelo
ner_model = pipeline("ner", model='monilouise/ner_pt_br', grouped_entities=False, device=env_GPU_CPU)
df_mercado['entities'] = df_mercado['text'].progress_apply(extract_entities)

#print(df_mercado.head(1))
#exit()

##------------------------------------------------------------------

#Contar a frequência das organizações
org_counter = Counter()
for entities in df_mercado['entities']:
    for entity in entities:
        org_counter[entity] += 1

#Criar um DataFrame com as organizações e suas contagens
org_rankings = pd.DataFrame(org_counter.items(), columns=['organizacao', 'total'])
org_rankings = org_rankings.sort_values(by='total', ascending=False)

#print(org_rankings)
#exit()

##------------------------------------------------------------------

#Gera um txt para ver o conteudo das organizações e identificar as falhas
org_rankings['organizacao'].to_csv(os.path.join(os.path.dirname(__file__), 'data','result_orgs.txt'), sep='\t', index=False, header=False)

##------------------------------------------------------------------

#Filtra as top 10 organizações
top_10_orgs = org_rankings.head(10)

#Plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='total', y='organizacao', data=top_10_orgs, palette='tab10')
plt.title('Top 10 Organizações - Mercado - 1° Trimestre de 2015')
plt.xlabel('Qtd de Aparições')
plt.ylabel('Organização')
for container in ax.containers:
    ax.bar_label(container)
plt.show()
