import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

df = pd.read_csv('C:/Users/12612/OneDrive/바탕 화면/I.D.E.A/데이터셋/02)웰니스_대화_스크립트_데이터셋.csv')
df = df.drop(columns=['임상질문그룹(연세의료원제공)', 'utterance(긍정)', 'utterance(부정)', '긍정에 대한 챗봇 답변', '부정에 대한 챗봇 답변', '추가발화(190917)', '추가발화 (191031)'])
df = df[~df['response(공감)'].isna()]

df['embedding'] = df['utterance(2차)'].map(lambda x: list(model.encode(x)))

text = input('In: ')
embedding = model.encode(text)

df['similarity'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())

print(df.head()) 

answer = df.loc[df['similarity'].idxmax()]

a = ['핵심증상', 'intent', '특이사항', '연관표현', 'utterance(2차)', 'response(공감)', 'similarity']
Output = ['핵심 증상', '구분', '특이사항', '연관표현', '유사한 발화', '응답', '유사도']

for i in range(7):
      print(Output[i] +":", str(answer[a[i]]) +"\n")