import os
import tensorflow as tf

# ==========================
# Path Configuration
# ==========================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.join(ROOT_DIR, "data_split")

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# ==========================
# Two Augmentation Pipeline (Only for Train)
# ==========================
# Baseline：轻度增强
baseline_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.02),
    tf.keras.layers.RandomZoom(0.05),
])


# 归一化函数
def normalize(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

# 数据集预处理通用逻辑
def prepare_dataset(dataset, training=False, aug_layer=None):
    # 归一化
    dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE)

    # 只有训练集才加入数据增强
    if training and aug_layer is not None:
        dataset = dataset.map(lambda x, y: (aug_layer(x), y), num_parallel_calls=AUTOTUNE)

    if training:
        dataset = dataset.shuffle(1000)

    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

# 统一入口函数，传入aug类型切换基线/增强
def load_dataset(aug_mode="baseline"):
    # 加载原始文件夹数据
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    # 选择对应增强层
    if aug_mode == "baseline":
        aug = baseline_augmentation
    else:
        raise ValueError("aug_mode only support baseline / improved")

    # 训练集绑定增强，val/test不传aug（无随机变换）
    train_ds = prepare_dataset(train_ds, training=True, aug_layer=aug)
    val_ds = prepare_dataset(val_ds, training=False)
    test_ds = prepare_dataset(test_ds, training=False)

    return train_ds, val_ds, test_ds

# ==========================
# Test Run
# ==========================
if __name__ == "__main__":
    # 测试基线轻度增强数据集
    print("===== Baseline Dataset (Weak Aug) =====")
    train_base, val, test = load_dataset(aug_mode="baseline")
    print(train_base)

    # 测试改进增强数据集
    print("\n===== Improved Dataset (Strong Aug) =====")
    train_imp, _, _ = load_dataset(aug_mode="improved")
    print(train_imp)