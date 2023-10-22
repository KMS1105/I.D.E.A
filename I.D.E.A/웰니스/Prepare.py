import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

df = pd.read_csv('C:/Users/user/Desktop/Coding/Python/python_AI/I.D.E.A/I.D.E.A-main/I.D.E.A-main/I.D.E.A/데이터셋/웰니스_대화_스크립트_데이터셋.csv')
df = df.drop(columns=['Unnamed: 3'])
df = df[~df['챗봇'].isna()]
df.to_csv("C:/Users/user/Desktop/Coding/Python/python_AI/I.D.E.A/I.D.E.A-main/I.D.E.A-main/I.D.E.A/데이터셋/ndf.csv", index=False)

df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

text = input('In: ')
embedding = model.encode(text)

df['similarity'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())

print(df.head()) 

answer = df.loc[df['similarity'].idxmax()]

a = ['구분', '유저', '챗봇']
Output = ['구분', '유사한 질문', '챗봇 대답']

for i in range(3):
      print(Output[i] +":", answer[a[i]] +"\n")
      
print(answer['similarity'])