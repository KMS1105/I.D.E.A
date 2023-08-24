import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

df = pd.read_csv('C:/Users/12612/OneDrive/바탕 화면/I.D.E.A/데이터셋/웰니스_대화_스크립트_데이터셋.csv')
df = df.drop(columns=['Unnamed: 3'])
df = df[~df['챗봇'].isna()]

df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

df.to_csv("C:/Users/12612/OneDrive/바탕 화면/I.D.E.A/데이터셋/Data preprocessing.csv", index=False)
