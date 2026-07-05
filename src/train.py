import tensorflow as tf
from model import build_cnn, build_mobilenet
from src.data_augmentation import load_dataset
import os

os.makedirs("models", exist_ok=True)


# ==========================
# TRAIN FUNCTION
# ==========================
def train_model(model_name, model, aug_mode):

    print("\n==============================")
    print(f"Running Experiment: {model_name}")
    print("==============================\n")

    train_ds, val_ds, test_ds = load_dataset(aug_mode=aug_mode)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=1,
        batch_size=16
    )

    test_loss, test_acc = model.evaluate(test_ds)

    print("\n===== TEST RESULT =====")
    print(model_name)
    print("Loss:", test_loss)
    print("Accuracy:", test_acc)

    model.save(f"models/{model_name}.keras")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    print("START TRAINING PIPELINE")

    # A: CNN + baseline
    cnn_model = build_cnn()
    train_model("A_CNN_baseline", cnn_model, "baseline")

    # B: MobileNet + baseline
    mobilenet_b = build_mobilenet()
    train_model("B_MobileNet_baseline", mobilenet_b, "baseline")
