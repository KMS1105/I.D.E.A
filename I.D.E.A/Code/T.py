import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

df = pd.read_csv('C:/Users/user/Desktop/Coding/Python/python_AI/I.D.E.A/I.D.E.A-main/I.D.E.A-main/I.D.E.A/데이터셋/02)웰니스_대화_스크립트_데이터셋.csv')
df = df.drop(columns=['특이사항','utterance','임상질문그룹(연세의료원제공)', 'utterance(긍정)', 'utterance(부정)', '긍정에 대한 챗봇 답변', '부정에 대한 챗봇 답변', '추가발화(190917)', '추가발화 (191031)'])

#0번째 행 & 마지막 행에 빈칸 추가
Nrow = pd.DataFrame(columns=df.columns)
Nrow.loc[0]=[0,0,0,0,0,0]
Ndf = pd.concat([Nrow, df, Nrow], ignore_index=True)
Ndf = Ndf.fillna(0)

def rp(l, col):
      nums = []
      nume = []
      numr = []
      numu = []
      
      #항 묶음 구하기
      for a in range(len(l.index)-1):
            if (l.iloc[a, col] == 0) & (l.iloc[a+1, col] != 0):
                nums.append(a+1)
                    
            if (l.iloc[a, col] != 0) & (l.iloc[a+1, col] == 0):
                nume.append(a+1)

            #print("l.iloc[{}, {}]: ".format(a, col)+str(l.iloc[a, col]))     

      print("nums: "+str(nums))
      print("nume: "+str(nume))

      #항 개수 구하기               
      for b in range(len(nums)):
            vnum = nume[b] - nums[b]
            numr.append(vnum)
      
      print("numr: "+str(numr))
            
      #항 구하기
      for c in range(len(numr)):
            k = []
            varl = nums[c] + 1
            k.append(varl)
            
            if numr[c] - 1 != 0:
                  for i in range(numr[c] - 1):
                        varl += 1
                        k.append(varl)
                        
            numu.append(k)
                  
            return numu

print(Ndf)
columns = ['핵심증상', 'intent', 'keyword(임상키워드)', '연관표현', 'response(공감)']
print(columns)

#묶음행에 빈칸이 있는 경우 다음 묶음행 전까지 묶음행을 반복해서 넣기
for x in range(len(columns)):
      repeat = rp(Ndf, x)
      rpnum1 = 0
      print("rp(Ndf, x): "+str(repeat))
      
      """for y in range(len(Ndf.index)):
            rpnum2 = 0
            
            if y == repeat[rpnum1+1][0]:
                  if rpnum1 <= len(repeat)-1:
                        rpnum1 += 1
                        
            if Ndf.iloc[y+1][x] == None:
                  if rpnum2 > repeat[rpnum1][-1]:
                        rpnum2 = 0
                        
                  Ndf.iloc[y+1][x] = Ndf.iloc[repeat[rpnum1][rpnum2]][x]
            
            rpnum2 += 1"""
            