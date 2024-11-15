import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from collections import Counter
import csv

def load_semantic_kitti_data(bin_path, label_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # x, y, z 좌표만 사용
    
    labels = np.fromfile(label_path, dtype=np.uint32)
    labels = labels & 0xFFFF  # 레이블은 하위 16비트에 저장됨
    
    return points, labels

def intersection_and_union(output, target, K, ignore_index=0):
    mask = (target != ignore_index)
    output = output[mask]
    target = target[mask]

    intersect = output[output == target]
    area_intersect = np.histogram(intersect, bins=K, range=(0, K-1))[0]
    area_output = np.histogram(output, bins=K, range=(0, K-1))[0]
    area_target = np.histogram(target, bins=K, range=(0, K-1))[0]
    area_union = area_output + area_target - area_intersect

    return area_intersect, area_union, area_target

# 경로 설정
data_root = Path("/media/jaemin/SSD8T/yerim/PTv3/dataset")
velodyne_dir = data_root / "sequences" / "06" / "velodyne"
labels_dir = data_root / "sequences" / "06" / "labels"
pred_labels_dir = Path("/home/yerim/Pointcept/exp/custom/custom-ptv3-epoch50-0828/result")
out_path = Path("/home/yerim/Pointcept/exp/custom/custom-ptv3-epoch50-0828/val_miou.csv")

all_preds = []
all_reals = []

with open(out_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame_num", "miou", "K", "entropy"])

    for velodyne_file in tqdm(sorted(velodyne_dir.glob('*.bin'))):
        frame_num = velodyne_file.stem
        label_file = labels_dir / f"{frame_num}.label"
        pred_label_file = pred_labels_dir / f"06_{frame_num}_pred.npy"

        # 데이터 로드
        _, real_labels = load_semantic_kitti_data(velodyne_file, label_file)
        pred_labels = np.load(pred_label_file)

        unique_labels = np.unique(real_labels)
        unique_K = len(unique_labels[unique_labels != 0])
        
        K = 20  # Semantic KITTI의 클래스 수에 맞게 조정
        intersection, union, _ = intersection_and_union(
            output=pred_labels,
            target=real_labels,
            K=K,
            ignore_index=0)
        
        ious = intersection / (union + 1e-10)
        miou = np.sum(ious) / unique_K

        entropy = stats.entropy(real_labels)
        writer.writerow([frame_num, f"{miou:.4f}", unique_K, f"{entropy:.4f}"])

        all_preds.extend(pred_labels.tolist())
        all_reals.extend(real_labels.tolist())

    # 전체 데이터셋에 대한 mIoU 계산
    intersection, union, _ = intersection_and_union(
        output=np.array(all_preds),
        target=np.array(all_reals),
        K=K,
        ignore_index=0)
    ious = intersection / (union + 1e-10)
    miou = np.mean(ious)

    writer.writerow([])
    writer.writerow(["Overall mIoU", f"{miou:.4f}"])
    writer.writerow([])

    # 라벨 분석
    unique_real_labels = np.unique(all_reals)
    unique_pred_labels = np.unique(all_preds)
    real_label_dist = Counter(all_reals)
    pred_label_dist = Counter(all_preds)

    writer.writerow(["Label Analysis"])
    writer.writerow(["Label", "Real Count", "Real Percentage", "Pred Count", "Pred Percentage", "IoU"])
    
    for label in sorted(set(unique_real_labels) | set(unique_pred_labels)):
        real_count = real_label_dist.get(label, 0)
        pred_count = pred_label_dist.get(label, 0)
        real_percentage = real_count / len(all_reals) * 100
        pred_percentage = pred_count / len(all_preds) * 100
        iou = ious[label] if label < len(ious) else 0
        
        writer.writerow([
            label,
            real_count,
            f"{real_percentage:.2f}%",
            pred_count,
            f"{pred_percentage:.2f}%",
            f"{iou:.4f}"
        ])

print(f"Analysis results saved to {out_path}")

# 라벨 불일치 확인
mismatched_labels = set(unique_pred_labels) - set(unique_real_labels)
if mismatched_labels:
    print("\nWarning: The following labels are in predictions but not in real labels:", mismatched_labels)

missing_labels = set(unique_real_labels) - set(unique_pred_labels)
if missing_labels:
    print("\nWarning: The following labels are in real labels but not in predictions:", missing_labels)