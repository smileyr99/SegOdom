import numpy as np

pred_labels = np.load("/home/yerim/Pointcept/exp/custom/custom-ptv3-epoch50-0828/result/06_000000_pred.npy")
print(pred_labels.shape)
print(np.unique(pred_labels))
print(pred_labels[:20])  # 처음 20개 예측값 출력