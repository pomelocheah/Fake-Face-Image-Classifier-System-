import tensorflow as tf

def mobilenet_preprocess(img):
    # 等价于官方preprocess_input：像素0~255 → [-1, 1]
    img = tf.cast(img, tf.float32)
    img /= 127.5
    img -= 1.0
    return img

# 验证集预处理直接调用自定义函数
def val_process(image, label):
    image = mobilenet_preprocess(image)
    return image, label

# 1. 配置参数
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# 2. 只加载验证集val，不碰train
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data_split/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=42
)

# 3. 验证集预处理（不做数据增强）
def val_process(image, label):
    image = mobilenet_preprocess(image)
    return image, label

val_ds = val_ds.map(val_process, tf.data.AUTOTUNE)

# 4. 加载你本地训练好的模型
model = tf.keras.models.load_model("models/B_MobileNet_baseline.keras")

# 5. 在验证集上评估，输出整体loss & accuracy
val_loss, val_acc = model.evaluate(val_ds, verbose=1)
print(f"\n==== Validation Set Result ====")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.2%}")

