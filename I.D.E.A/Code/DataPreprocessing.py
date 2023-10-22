import pandas as pd

df = pd.read_csv('C:/Users/user/Desktop/Coding/Python/python_AI/I.D.E.A/I.D.E.A-main/I.D.E.A-main/I.D.E.A/데이터셋/02)웰니스_대화_스크립트_데이터셋.csv')
df = df.drop(columns=['특이사항','utterance','임상질문그룹(연세의료원제공)', 'utterance(긍정)', 'utterance(부정)', '긍정에 대한 챗봇 답변', '부정에 대한 챗봇 답변', '추가발화(190917)', '추가발화 (191031)'])
df = df[~df['response(공감)'].isna()]

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

            for d in range(numr[c]):
                  varl = (nums[c]+d)
                  k.append(varl)

            numu.append(k)

      #print("numu: ",numu)

      return numu

print(Ndf)

#묶음행에 빈칸이 있는 경우 다음 묶음행 전까지 묶음행을 반복해서 넣기
for x in range(len(df.columns)):
      repeat = rp(Ndf, x)
      rpnum1 = 0 #밖 순차
      rpnum2 = -1 #안 순차
      
      for y in range(len(Ndf.index)):
            if rpnum1 < len(repeat)-1:
                  if (y == repeat[rpnum1+1][0]): #밖 다음 순차
                    rpnum1 += 1
                    rpnum2 = -1
                              
                  if Ndf.iloc[y+1][x] == 0:
                        rw = int(y+1)
                        Nrw = int(repeat[rpnum1][rpnum2])

                        Ndf.iloc[rw][x] = Ndf.iloc[Nrw][x]
                        
                        if repeat[rpnum1][rpnum2] == repeat[rpnum1][-1]:
                              rpnum2 = 0
                        
                        e = repeat[rpnum1][-1] 
                        s = repeat[rpnum1][0]
                        
                        if (e - s) != 0: #안 다음 순차
                              rpnum2 += 1

Ndf.to_csv("C:/Users/user/Desktop/Coding/Python/python_AI/I.D.E.A/I.D.E.A-main/I.D.E.A-main/I.D.E.A/데이터셋/Nndf.csv", index=False)
print(Ndf)