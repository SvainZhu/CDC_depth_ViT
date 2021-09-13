import math
import numpy as np

def eval_state(probs, labels, thr):
    predict = []
    for prob in probs:
        pred = prob >= thr
        predict.append(pred)
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def calculate_statistic(probs, labels):
    TN, FN, FP, TP = eval_state(probs, labels, 0.5)
    APCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)
    NPCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)
    ACER = (APCER + NPCER) / 2.0
    ACC = (TP + TN) / labels.shape[0]
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif(FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    return APCER, NPCER, ACER, ACC, HTER

