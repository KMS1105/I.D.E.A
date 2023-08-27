
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('C:/Users/12612/OneDrive/바탕 화면/I.D.E.A/데이터셋/02)웰니스_대화_스크립트_데이터셋.csv')
Ndf = df.fillna('none')
print(Ndf.head())

def T(l, col):
    numS = []
    
    for a in range(len(l.index)):
        if (l.iloc[a][col] != 'none') & (l.iloc[a+1][col] == 'none'):
            numS.append(a)
            
    print(numS)
    
T(Ndf, '핵심증상')