import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def eval_state(probs, labels, thr):
    labels, probs = np.array(labels), np.array(probs)
    TN, FP, FN, TP = confusion_matrix(labels, np.where(probs > thr, 1, 0)).ravel()
    return TN, FN, FP, TP

def calculate_statistic(probs, labels):
    TN, FN, FP, TP = eval_state(probs, labels, 0.5)
    APCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)
    NPCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)
    ACER = (APCER + NPCER) / 2.0
    ACC = (TP + TN) / len(labels)
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
