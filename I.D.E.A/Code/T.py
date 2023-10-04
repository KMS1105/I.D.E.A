import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

df = pd.read_csv('C:/Users/12612/OneDrive/바탕 화면/학교/I.D.E.A/데이터셋/02)웰니스_대화_스크립트_데이터셋.csv')
df = df.drop(columns=['임상질문그룹(연세의료원제공)', 'utterance(긍정)', 'utterance(부정)', '긍정에 대한 챗봇 답변', '부정에 대한 챗봇 답변', '추가발화(190917)', '추가발화 (191031)'])
df = df[~df['response(공감)'].isna()]

#0번째 행 & 마지막 행에 빈칸 추가
new_row = []

for i in range(len(df.columns)):
      new_row.append('')
      
Nrow = pd.DataFrame(new_row)
Ndf = pd.concat([df.iloc[:0], Nrow, df.iloc[0:]], ignore_index=True)
Ndf = pd.concat([Nrow], ignore_index=True)

def rp(l, col):
      n = None
      nums = []
      nume = []
      numr = []
      numu = []
      
      #항 묶음 구하기
      for a in range(len(l.index)):
            if (l.iloc[a, l.columns[col]].isna() == False) & (l.iloc[a+1, l.columns[col]].isna() == True):
                  nums.append(a)
                        
            if (l.iloc[a, l.columns[col]].isna() == True) & (l.iloc[a+1, l.columns[col]].isna == False):
                  nume.append(a)

      #항 개수 구하기               
      for b in range(len(nums)):
            Vnum = nume[b] - nums[b]
            numr.append(Vnum)
            
      #항 구하기
      for c in range(len(numr)):
            k = []
            varl = nums[c] + 1
            k.append(varl)
            
            if numr[c] - 1 != 0:
                  for x in range(numr[c] - 1):
                        varl += 1
                        k.append(varl)
                        
            numu.append(k)
                  
            return numu

#묶음행에 빈칸이 있는 경우 다음 묶음행 전까지 묶음행을 반복해서 넣기
for x in Ndf.columns:
      repeat = rp(Ndf, [x])
      rpnum1 = 0
      
      for y in range(len(Ndf.index)):
            rpnum2 = 0
            
            if y == repeat[rpnum1+1][0]:
                  if rpnum1 <= len(repeat)-1:
                        rpnum1 += 1
                        
            if Ndf.iloc[y+1][x] == None:
                  if rpnum2 > repeat[rpnum1][-1]:
                        rpnum2 = 0
                        
                  Ndf.iloc[y+1][x] = Ndf.iloc[repeat[rpnum1][rpnum2]][x]
            
            rpnum2 += 1
            
print(Ndf.head())