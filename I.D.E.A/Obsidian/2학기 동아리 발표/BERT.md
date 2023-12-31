[[목차]]
[[Sentence Bert]]

BERT(Bidirectional Encoder Representations from Transformers)는 2018년에 구글이 공개한 사전 훈련된 모델입니다. 
수많은 [[NLP]] 태스크에서 최고 성능을 보여주면서 명실공히 [[NLP]]의 한 획을 그은 모델로 평가받고 있습니다. 

BERT는 위키피디아(25억 단어)와 BooksCorpus(8억 단어)와 같은 레이블이 없는 텍스트 데이터로 사전 훈련된 언어 모델입니다.

BERT가 높은 성능을 얻을 수 있었던 것은, 레이블이 없는 방대한 데이터로 사전 훈련된 모델을 가지고, 레이블이 있는 다른 작업(Task)에서 추가 훈련과 함께 하이퍼파라미터를 재조정하여 이 모델을 사용하면 성능이 높게 나오는 기존의 사례들을 참고하였기 때문입니다. 다른 작업에 대해서 파라미터 재조정을 위한 추가 훈련 과정을 파인 튜닝(Fine-tuning)이라고 합니다.

위의 그림은 BERT의 파인 튜닝 사례를 보여줍니다. 우리가 하고 싶은 태스크가 스팸 메일 분류라고 하였을 때, 이미 위키피디아 등으로 사전 학습된 BERT 위에 분류를 위한 신경망을 한 층 추가합니다. 이 경우, 비유하자면 BERT가 언어 모델 사전 학습 과정에서 얻은 지식을 활용할 수 있으므로 스팸 메일 분류에서 보다 더 좋은 성능을 얻을 수 있습니다.