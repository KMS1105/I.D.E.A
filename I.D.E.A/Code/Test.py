"""a = ['구분', '유저', '챗봇', 'similarity']
Output = ['구분', '유사한 질문', '챗봇 대답', '유사도']

for i in range(4):
      print(Output[i]+":", a[i]+"\n")"""
      
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

df = pd.read_csv('C:/Users/12612/OneDrive/바탕 화면/I.D.E.A/데이터셋/02)웰니스_대화_스크립트_데이터셋.csv')
df = df.drop(columns=['임상질문그룹(연세의료원제공)', 'utterance(긍정)', 'utterance(부정)', '긍정에 대한 챗봇 답변', '부정에 대한 챗봇 답변', '추가발화(190917)', '추가발화 (191031)'])
df = df[~df['response(공감)'].isna()]

#0번째 행 & 마지막 행에 빈칸 추가
new_row = []

for i in range(len(df.columns)):
      new_row.append('')
      
Nrow = pd.DataFrame(new_row)
Ndf = pd.concat([df.iloc[:0], Nrow, df.iloc[0:]], ignore_index=True)
Ndf = pd.concat([Nrow], ignore_index=True)

def RP(l, col):
      numS = []
      numE = []
      numR = []
      numU = []
      
      for a in range(len(l.index)):
            if (l[a][l.columns[col]] != None) & (l[a+1][l.columns[col]] == None):
                  numS.append(a)
                        
            if (l[a][l.columns[col]] == None) & (l[a+1][l.columns[col]] != None):
                  numE.append(a)
                        
      for b in range(len(numS)):
            Vnum = numE[b] - numS[b]
            numR.append(Vnum)
            
      for c in range(len(numR)):
            k = []
            varl = numS[i] + 1
            k.append(varl)
            
            if numR[i] - 1 != 0:
                  for x in range(numR[i] - 1):
                        varl += 1
                        k.append(varl)
                        
                  numU.append(k)
                  
            return numU 
      
for x in Ndf.columns:
      repeat = RP(Ndf, [x])
      rpnum1 = 0
      
      for y in range(len(Ndf.index)):
            rpnum2 = 0
            
            if y == repeat[rpnum1+1][0]:
                  if rpnum1 <= len(repeat)-1:
                        rpnum1 += 1
                        
            if Ndf.iloc[y+1][x] == None:
                  if rpnum2 > repeat[rpnum1][-1]:
                        rpnum2 = 0
                        
                  Ndf.iloc[y+1][x] = Ndf.iloc[repeat[rpnum1][rpnum2]][a]
            
            rpnum2 += 1
            
print(Ndf.head())