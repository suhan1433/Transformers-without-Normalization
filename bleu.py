from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu(references, predictions):
    """
    BLEU 점수를 계산하는 함수.
    :param predictions: 모델 예측 결과 리스트 (각각의 예측된 번역 문장들)
    :param references: 실제 정답 리스트 (각각의 실제 번역 문장들)
    :return: BLEU 점수
    """
    
    return corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25))



# references = [['Eine', 'Gruppe', 'von', 'Männern', 'lädt', 'Baumwolle', 'auf', 'einen', 'Lastwagen'], ["하이"] ]
# predictions = [['Eine', 'Gruppe', 'von', 'Männern', 'lädt', 'sich', 'auf', 'einen', 'Lkw.....'], ['하이']]

# print(compute_bleu(references, predictions))