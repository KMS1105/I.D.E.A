import pandas as pd
from numpy import dot
from numpy.linalg import norm
import numpy as np
from sentence_transformers import SentenceTransformer

def cos_sim(A, B):
      return dot(A, B)/(norm(A)*norm(B))

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

df = pd.read_csv('C:/Users/12612/OneDrive/바탕 화면/I.D.E.A/데이터셋/웰니스_대화_스크립트_데이터셋.csv')
df = df.drop(columns=['Unnamed: 3'])
df = df[~df['챗봇'].isna()]

df.loc[0, '유저']

df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

print(df.head())

"""Text = input('In: ')
embedding = model.encode(Text)

df['similarity'] = df['embedding'].map(lambda x: cos_sim([embedding], [x]).squeeze())

print(df.head())"""