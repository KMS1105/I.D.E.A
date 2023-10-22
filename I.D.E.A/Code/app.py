import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

st.title("감정 분석")

df = pd.read_csv('C:/Users/user/Desktop/Coding/Python/python_AI/I.D.E.A/I.D.E.A-main/I.D.E.A-main/I.D.E.A/데이터셋/Nndf.csv')
#df = df.drop(columns=['임상질문그룹(연세의료원제공)', 'utterance(긍정)', 'utterance(부정)', '긍정에 대한 챗봇 답변', '부정에 대한 챗봇 답변', '추가발화(190917)', '추가발화 (191031)'])
#df = df[~df['response(공감)'].isna()]

df['embedding'] = df['utterance(2차)'].map(lambda x: list(model.encode(x)))

with st.form("form"):
    User_input = st.text_input("Prompt")
    submit = st.form_submit_button("Submit")

if submit and User_input:
      text = User_input
      embedding = model.encode(text)

      df['similarity'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
      answer = df.loc[df['similarity'].idxmax()]

      a = ['intent', '연관표현', 'utterance(2차)', 'response(공감)', 'similarity']
      Output = ['구분', '연관표현', '유사한 발화', '응답', '유사도']

      for i in range(len(a)):
            st.write(Output[i] +":", str(answer[a[i]]),"\n")
