import pandas as pd
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache_resource
def cashed_model():
     model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
     return model

@st.cache_data
def get_dataset():
     df = pd.read_csv('C:/Users/user/Desktop/Coding/Python/python_AI/I.D.E.A/I.D.E.A-main/I.D.E.A-main/I.D.E.A/데이터셋/Nndf.csv')
     df['embedding'] = df['utterance(2차)'].apply(json.loads)
     return df

model = cashed_model()
df = get_dataset()

st.header("감정 분석")

if 'generated' not in st.session_state:
     st.session_state['generated'] = []

if 'past' not in st.session_state:
     st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
     User_input = st.text_input("You: ", '')
     submit = st.form_submit_button("Submit")

if submit and User_input:
      embedding = model.encode(User_input)

      df['similarity'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
      answer = df.loc[df['similarity'].idxmax()]

      a = ['intent', '연관표현', 'utterance(2차)', 'response(공감)', 'similarity']
      #Output = ['구분', '연관표현', '유사한 발화', '응답', '유사도']
      st.session_state.past.append(User_input)
      
      for i in range(len(a)):
            st.session_state.past.append(answer[a[i]])
            #st.write(Output[i] +":", str(answer[a[i]]),"\n")

for i in range(len(st.session_state['past'])):
     message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')
     
     if len(st.session_state['generated']) > i:
          message(st.session_state['past'][i], is_user=True, key=str(i)+'_bot')