import tensorflow as tf
from model import build_cnn, build_mobilenet
from src.data_augmentation import load_dataset
import os
import mlflow
import mlflow.tensorflow
from config import EPOCHS, IMAGE_SIZE, BATCH_SIZE

os.makedirs("models", exist_ok=True)

mlflow.set_experiment("Fake_Real_Face_Classification")
mlflow.tensorflow.autolog()

# ==========================
# TRAIN FUNCTION
# ==========================
def train_model(model_name, model, aug_mode):

    print("\n==============================")
    print(f"Running Experiment: {model_name}")
    print("==============================\n")

    train_ds, val_ds, test_ds = load_dataset(aug_mode=aug_mode)
    
    with mlflow.start_run(run_name=model_name):
        # 手动记录实验超参
        mlflow.log_param("augmentation_mode", aug_mode)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("input_image_size", f"{IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=1,
        batch_size=16
    )

    test_loss, test_acc = model.evaluate(test_ds)

    # 记录测试结果
    with mlflow.start_run(run_name=model_name):
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

    print("\n===== TEST RESULT =====")
    print(model_name)
    print("Loss:", test_loss)
    print("Accuracy:", test_acc)

    model.save(f"models/{model_name}.keras")
    mlflow.log_artifact(f"models/{model_name}.keras", artifact_path="saved_model")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    print("START TRAINING PIPELINE")

    # A: CNN + baseline
    cnn_model = build_cnn()
    train_model("A_CNN_baseline", cnn_model, "baseline")
    mlflow.end_run()

    # B: MobileNet + baseline
    mobilenet_b = build_mobilenet()
    train_model("B_MobileNet_baseline", mobilenet_b, "baseline")
    mlflow.end_run()
