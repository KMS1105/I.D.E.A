import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
print("데이터셋 파일 경로 예시\nC:/Users/user/Desktop/Coding/Python/python_AI/I.D.E.A/I.D.E.A-main/I.D.E.A-main/I.D.E.A/데이터셋/Nndf.csv\n[[\를 /로 바꿔줘야 합니다.]]") 
fpath = input("데이터셋 파일 경로: ")
df = pd.read_csv(fpath)
df['embedding'] = df['utterance(2차)'].map(lambda x: list(model.encode(x)))

print("전처리 과정에서 데이터가 많이 삭제되어 정확도가 높진 않습니다.")

while True:
    text = input("In: ")

    embedding = model.encode(text)

    df['similarity'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['similarity'].idxmax()]

    a = ['intent', '연관표현', 'utterance(2차)', 'response(공감)', 'similarity']
    Output = ['구분', '연관표현', '유사한 발화', '응답', '유사도']

    for i in range(len(a)):
        print(Output[i] +":", str(answer[a[i]]),"\n")