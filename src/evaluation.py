import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Robust import for load_model: prefer tensorflow.keras but fall back to standalone keras
try:
    from tensorflow.keras.models import load_model
except Exception:
    try:
        from keras.models import load_model
    except Exception:
        import tensorflow as tf
        load_model = tf.keras.models.load_model

def evaluate_model(model, test_ds, model_name="model"):
    y_true = []
    y_pred = []

    # 遍历测试集获取真实标签与预测结果
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        preds = (preds > 0.5).astype(int)
        y_true.extend(labels.numpy().astype(int))
        y_pred.extend(preds.flatten())

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("\n============================")
    print(f"{model_name} CONFUSION MATRIX")
    print("============================")
    print(cm)

    # 分类报告 修正标签顺序 real=0, fake=1
    report = classification_report(
        y_true,
        y_pred,
        target_names=["real", "fake"],
        output_dict=True
    )

    accuracy = report["accuracy"]
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    print("\n===== METRICS =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["real", "fake"])
    plt.yticks(tick_marks, ["real", "fake"])

    # 填充数字
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.show()

    return accuracy, precision, recall, f1

# ---------------------- 测试入口，必须加！不然直接运行无输出 ----------------------
if __name__ == "__main__":
    from data_augmentation import load_dataset
    # 1. 加载测试集
    _, _, test_ds = load_dataset(aug_mode="baseline")
    # 2. 加载你训练好的模型，修改路径为你的模型文件
    model = load_model("models/A_CNN_baseline.keras")
    # 3. 执行评估
    acc, prec, rec, f1score = evaluate_model(model, test_ds, model_name="Baseline CNN")