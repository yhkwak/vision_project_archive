# 논문 검색 (2024.12.05)
- 딥러닝 기법을 활용한 프로브카드 외관 불량 탐지에 관한 연구
- 지도학습 기반 불량 탐지 모델을 위한 능동학습 초기화 방법론
- 객체 탐지 기술을 통한 휠 너트 제품의 단조 공정에서 불량 검출
<br><br>
------------
<br><br>

# 사전 연습
- AI Hub의 부품 품질 검사 영상 데이터(선박·해양플랜드) (고도화) - LNG탱크 품질 검사 영상 데이터 중 조인트 용접관련 불량 데이터 이용하여 CNN을 이용한 불량 검사 테스트 (2024.12.05) -- ~~Cuda 이용한 학습 필요 (시간이 오래 걸려 제대로 작동하는지 확인 불가능)~~ 학습 확인. 제대로 작동하는지는 확인 필요
  
![image](https://github.com/user-attachments/assets/acb60f45-fd9c-4ad2-aacb-042544655696)

2시간 동안 2epoch 돌렸는데 첫 epoch부터 정확도가 너무 높게 나와서 학습이 잘 된건지 의문이다.

- Yolo v8을 사용할지 CNN을 사용할지 고려, 하이브리드 모델도 사용 가능

- 다른 데이터셋을 이용하여 다시 학습
<br>

![image](https://github.com/user-attachments/assets/de074435-e2db-49c2-a4ad-c3182c65719b)

처음엔 train 94%, validation 7% 나오다가 어느순간 validation 값 급격하게 상승
test 데이터도 최종 100%

사용 데이터: 품질 이상탐지,진단(크로메이트) AI 데이터셋
train: pass 416장, fail 59장
validation: pass 148장, fail 12장
test: pass 128장, fail 3장


