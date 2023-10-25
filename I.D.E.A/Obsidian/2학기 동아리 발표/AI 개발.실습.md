[[목차]]
[[메인]]

기본적인 틀과 비슷하게 코드를 짰습니다.
Sentence Bert를 사용하기 위해서 SentenceTransformer모델을 불러오고, 전 처리된 파일을 불러온뒤 'utterance(2차)'(발화 데이터)를 SentenceTransformer로 인코딩하여 'embedding' 이라는 새로운 칼럼에 넣습니다. 
이후 측정할 인풋 데이터 역시 모델로 인코딩하고, 'similarity' 칼럼을 추가하여 코사인 유사도를 통해 유사도를 측정합니다. 
'embedding'의 발화 데이터와 인풋 데이터의 유사도 값이 가장 높은 발화 데이터가 들어있는 행을 출력합니다.
