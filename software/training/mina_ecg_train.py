import os
from pathlib import Path

import wfdb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# =========================================================
# 1. CẤU HÌNH PROJECT
# =========================================================

PROJECT_NAME = "MINA_ECG_1D_CNN_MAINPDF"

BASE_DIR = Path(__file__).resolve().parent

# Dataset local đặt cùng cấp với file Python
LOCAL_DATASET_DIR = BASE_DIR / "dataset"

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

for folder in [PROCESSED_DIR, MODEL_DIR, RESULT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. CẤU HÌNH DỮ LIỆU ECG THEO MAIN.PDF
# =========================================================

CLASS_NAMES = ["NOR", "LBBB", "RBBB", "PVC", "APB"]

LABEL_MAP = {
    "N": 0,   # NOR  - normal beat
    "L": 1,   # LBBB - left bundle branch block beat
    "R": 2,   # RBBB - right bundle branch block beat
    "V": 3,   # PVC  - premature ventricular contraction
    "A": 4,   # APB  - atrial premature beat
}

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209",
    "210", "212", "213", "214", "215", "217", "219", "220",
    "221", "222", "223", "228", "230", "231", "232", "233", "234"
]

# main.pdf dùng 46 recordings.
# MIT-BIH có 48 recordings, thường loại 102 và 104 do paced beat.
EXCLUDED_RECORDS = {"102", "104"}
RECORDS = [r for r in MITDB_RECORDS if r not in EXCLUDED_RECORDS]

LEFT_SAMPLES = 159
RIGHT_SAMPLES = 160
SEGMENT_LENGTH = LEFT_SAMPLES + 1 + RIGHT_SAMPLES   # 320 mẫu

RANDOM_STATE = 42
BATCH_SIZE = 20
EPOCHS = 15

LR_1 = 1e-4
LR_2 = 1e-5


# =========================================================
# 3. KIỂM TRA DATASET LOCAL
# =========================================================

def find_dataset_dir():
    """
    Tìm thư mục chứa file MIT-BIH local.
    Ưu tiên:
        1. ./dataset/
        2. ./dataset/dataset/
        3. ./data/raw_mitbih/
    """

    candidates = [
        BASE_DIR / "dataset",
        BASE_DIR / "dataset" / "dataset",
        BASE_DIR / "data" / "raw_mitbih",
        BASE_DIR,
    ]

    best_dir = None
    best_count = 0

    for folder in candidates:
        if not folder.exists():
            continue

        dat_files = list(folder.glob("*.dat"))
        standard_dat = [p for p in dat_files if p.stem.isdigit()]

        if len(standard_dat) > best_count:
            best_count = len(standard_dat)
            best_dir = folder

    if best_dir is None or best_count == 0:
        raise FileNotFoundError(
            "\n[ERROR] Không tìm thấy dataset MIT-BIH local.\n"
            "Hãy đặt dữ liệu theo dạng:\n\n"
            "MINA_ECG_1D_CNN_MAINPDF/\n"
            "├── mina_ecg_train.py\n"
            "└── dataset/\n"
            "    ├── 100.dat\n"
            "    ├── 100.hea\n"
            "    ├── 100.atr\n"
            "    └── ...\n"
        )

    print(f"[OK] Tìm thấy dataset local tại: {best_dir}")
    print(f"[OK] Số file .dat chuẩn tìm thấy: {best_count}")
    return best_dir


def check_record_files(dataset_dir, rec):
    """
    Kiểm tra 1 record có đủ .dat, .hea, .atr không.
    """
    dat = dataset_dir / f"{rec}.dat"
    hea = dataset_dir / f"{rec}.hea"
    atr = dataset_dir / f"{rec}.atr"

    return dat.exists() and hea.exists() and atr.exists()


# =========================================================
# 4. TIỀN XỬ LÝ ECG
# =========================================================

def extract_ecg_beats(dataset_dir):
    """
    Đọc ECG từ dataset local.
    Cắt mỗi beat:
        159 mẫu trước R-peak
        1 mẫu tại R-peak
        160 mẫu sau R-peak

    Output:
        X: (num_beats, 320, 1)
        y: (num_beats,)
    """

    X = []
    y = []

    print("\n================ BẮT ĐẦU ĐỌC DATASET ================")

    for rec in RECORDS:
        if not check_record_files(dataset_dir, rec):
            print(f"[SKIP] Record {rec}: thiếu .dat/.hea/.atr")
            continue

        record_path = str(dataset_dir / rec)

        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, "atr")
        except Exception as e:
            print(f"[WARNING] Không đọc được record {rec}: {e}")
            continue

        # Lấy kênh ECG đầu tiên
        signal = record.p_signal[:, 0].astype(np.float32)

        # Chuẩn hóa theo từng record
        signal_mean = np.mean(signal)
        signal_std = np.std(signal) + 1e-8
        signal = (signal - signal_mean) / signal_std

        beat_count = 0

        for r_peak, symbol in zip(annotation.sample, annotation.symbol):
            if symbol not in LABEL_MAP:
                continue

            start = r_peak - LEFT_SAMPLES
            end = r_peak + RIGHT_SAMPLES + 1

            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end]

            if len(beat) != SEGMENT_LENGTH:
                continue

            X.append(beat.reshape(SEGMENT_LENGTH, 1))
            y.append(LABEL_MAP[symbol])
            beat_count += 1

        print(f"[OK] Record {rec}: lấy được {beat_count} beat hợp lệ")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    if len(X) == 0:
        raise RuntimeError(
            "[ERROR] Không trích được beat nào. "
            "Hãy kiểm tra file .atr và nhãn N/L/R/V/A trong dataset."
        )

    print("\n================ THỐNG KÊ DATASET ================")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    for idx, name in enumerate(CLASS_NAMES):
        print(f"{name}: {(y == idx).sum()} mẫu")

    return X, y


def split_and_save_dataset(X, y):
    """
    Chia 70% train, 15% validation, 15% test.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print("\n[OK] Đã lưu dữ liệu processed vào data/processed/")

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_or_create_dataset():
    """
    Nếu đã có dữ liệu processed thì nạp lại.
    Nếu chưa có thì đọc từ dataset local.
    """

    processed_files = [
        PROCESSED_DIR / "X_train.npy",
        PROCESSED_DIR / "y_train.npy",
        PROCESSED_DIR / "X_val.npy",
        PROCESSED_DIR / "y_val.npy",
        PROCESSED_DIR / "X_test.npy",
        PROCESSED_DIR / "y_test.npy",
    ]

    if all(p.exists() for p in processed_files):
        print("[OK] Đã có dữ liệu processed. Nạp lại từ data/processed/")

        X_train = np.load(PROCESSED_DIR / "X_train.npy")
        y_train = np.load(PROCESSED_DIR / "y_train.npy")
        X_val = np.load(PROCESSED_DIR / "X_val.npy")
        y_val = np.load(PROCESSED_DIR / "y_val.npy")
        X_test = np.load(PROCESSED_DIR / "X_test.npy")
        y_test = np.load(PROCESSED_DIR / "y_test.npy")

        return X_train, X_val, X_test, y_train, y_val, y_test

    dataset_dir = find_dataset_dir()
    X, y = extract_ecg_beats(dataset_dir)
    return split_and_save_dataset(X, y)


# =========================================================
# 5. INCEPTION BLOCK THEO FIG. 4 MAIN.PDF
# =========================================================

def inception_block(x, filters, name):
    """
    Inception Block 1D:

        Input
        ├── MaxPool J=3 → Conv1D J=1
        └── Conv1D J=1 → Conv1D J=3
                       → Conv1D J=5
                       → Conv1D J=7
        → Concatenate
        → ReLU

    filters = N output channels.
    Mỗi nhánh chính ra N/4 channels.
    """

    branch_channels = filters // 4

    # Nhánh A: MaxPool J=3 -> Conv J=1
    branch_pool = tf.keras.layers.MaxPooling1D(
        pool_size=3,
        strides=1,
        padding="same",
        name=f"{name}_maxpool_J3"
    )(x)

    branch_pool = tf.keras.layers.Conv1D(
        filters=branch_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
        name=f"{name}_convJ1_1_after_pool"
    )(branch_pool)

    # Nhánh B: bottleneck Conv J=1
    bottleneck = tf.keras.layers.Conv1D(
        filters=branch_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
        name=f"{name}_convJ1_2_bottleneck"
    )(x)

    branch_j3 = tf.keras.layers.Conv1D(
        filters=branch_channels,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=True,
        name=f"{name}_convJ3"
    )(bottleneck)

    branch_j5 = tf.keras.layers.Conv1D(
        filters=branch_channels,
        kernel_size=5,
        strides=1,
        padding="same",
        use_bias=True,
        name=f"{name}_convJ5"
    )(bottleneck)

    branch_j7 = tf.keras.layers.Conv1D(
        filters=branch_channels,
        kernel_size=7,
        strides=1,
        padding="same",
        use_bias=True,
        name=f"{name}_convJ7"
    )(bottleneck)

    out = tf.keras.layers.Concatenate(
        axis=-1,
        name=f"{name}_concatenate"
    )([branch_pool, branch_j3, branch_j5, branch_j7])

    out = tf.keras.layers.ReLU(name=f"{name}_relu")(out)

    return out


# =========================================================
# 6. MINI INCEPTIONNET THEO MAIN.PDF
# =========================================================

def build_mini_inceptionnet():
    """
    Cấu trúc mạng:

        Input 320×1
        Conv1D J=7, stride=2, 8 channels
        2× Inception Block + Residual

        Conv1D J=5, stride=2, 16 channels
        2× Inception Block + Residual

        Conv1D J=3, stride=2, 32 channels
        2× Inception Block + Residual

        Global Average Pooling
        Linear Softmax 5 lớp
    """

    inputs = tf.keras.Input(
        shape=(SEGMENT_LENGTH, 1),
        name="ecg_input_320x1"
    )

    # 320×1 -> 160×8
    x = tf.keras.layers.Conv1D(
        filters=8,
        kernel_size=7,
        strides=2,
        padding="same",
        use_bias=True,
        name="conv1d_0_J7_stride2"
    )(inputs)
    x = tf.keras.layers.ReLU(name="relu_conv1d_0")(x)

    # 160×8 -> 160×8
    shortcut = x
    x = inception_block(x, filters=8, name="inception_1")
    x = inception_block(x, filters=8, name="inception_2")
    x = tf.keras.layers.Add(name="residual_add_1")([shortcut, x])

    # 160×8 -> 80×16
    x = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=5,
        strides=2,
        padding="same",
        use_bias=True,
        name="conv1d_1_J5_stride2"
    )(x)
    x = tf.keras.layers.ReLU(name="relu_conv1d_1")(x)

    # 80×16 -> 80×16
    shortcut = x
    x = inception_block(x, filters=16, name="inception_3")
    x = inception_block(x, filters=16, name="inception_4")
    x = tf.keras.layers.Add(name="residual_add_2")([shortcut, x])

    # 80×16 -> 40×32
    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=True,
        name="conv1d_2_J3_stride2"
    )(x)
    x = tf.keras.layers.ReLU(name="relu_conv1d_2")(x)

    # 40×32 -> 40×32
    shortcut = x
    x = inception_block(x, filters=32, name="inception_5")
    x = inception_block(x, filters=32, name="inception_6")
    x = tf.keras.layers.Add(name="residual_add_3")([shortcut, x])

    # 40×32 -> 1×32
    x = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(x)

    # 1×32 -> 5 lớp
    outputs = tf.keras.layers.Dense(
        units=len(CLASS_NAMES),
        activation="softmax",
        name="linear_softmax_5_classes"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Mini_InceptionNet_ECG"
    )

    return model


# =========================================================
# 7. CHI PHÍ TÍNH TOÁN
# =========================================================

def conv1d_mac(Y, N, K, J):
    """
    MAC = Y × N × K × J
    1 MAC ≈ 1 phép nhân + 1 phép cộng tích lũy
    """
    return Y * N * K * J


def inception_mac(Y, K, N):
    """
    Inception Block theo bài báo:

        ConvJ1.1: K   -> N/4, J=1
        ConvJ1.2: K   -> N/4, J=1
        ConvJ3:   N/4 -> N/4, J=3
        ConvJ5:   N/4 -> N/4, J=5
        ConvJ7:   N/4 -> N/4, J=7
    """

    c = N // 4

    mac_j1_1 = conv1d_mac(Y, c, K, 1)
    mac_j1_2 = conv1d_mac(Y, c, K, 1)
    mac_j3 = conv1d_mac(Y, c, c, 3)
    mac_j5 = conv1d_mac(Y, c, c, 5)
    mac_j7 = conv1d_mac(Y, c, c, 7)

    return mac_j1_1 + mac_j1_2 + mac_j3 + mac_j5 + mac_j7


def print_compute_cost():
    rows = []

    rows.append(("Conv1D-0 J7 320×1 -> 160×8", conv1d_mac(160, 8, 1, 7)))
    rows.append(("2×Inception 160×8", 2 * inception_mac(160, 8, 8)))
    rows.append(("Conv1D-1 J5 160×8 -> 80×16", conv1d_mac(80, 16, 8, 5)))
    rows.append(("2×Inception 80×16", 2 * inception_mac(80, 16, 16)))
    rows.append(("Conv1D-2 J3 80×16 -> 40×32", conv1d_mac(40, 32, 16, 3)))
    rows.append(("2×Inception 40×32", 2 * inception_mac(40, 32, 32)))
    rows.append(("Linear 32 -> 5", 32 * 5))

    total = 0

    print("\n================ CHI PHÍ TÍNH TOÁN ================")

    for name, mac in rows:
        total += mac
        print(f"{name:<35}: {mac:>10,} MAC")

    print("---------------------------------------------------")
    print(f"Tổng xấp xỉ                       : {total:>10,} MAC")
    print("1 MAC ≈ 1 phép nhân + 1 phép cộng tích lũy")
    print("===================================================\n")


# =========================================================
# 8. TRAINING
# =========================================================

def lr_schedule(epoch, lr):
    """
    main.pdf:
        learning rate = 0.0001 cho 40 epoch đầu
        sau đó giảm xuống 0.00001
    """
    if epoch < 40:
        return LR_1
    return LR_2


def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_mina_mini_inceptionnet.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return history


# =========================================================
# 9. ĐÁNH GIÁ MODEL
# =========================================================

def plot_training_curve(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "training_curve.png", dpi=300)
    plt.close()

    print("[OK] Đã lưu results/training_curve.png")


def calculate_metrics_from_cm(cm):
    """
    Tính ACC, SEN, SPEC, PPV cho từng lớp.
    """

    total = np.sum(cm)
    lines = []

    lines.append("Class,TP,TN,FP,FN,ACC,SEN,SPEC,PPV")

    for i, class_name in enumerate(CLASS_NAMES):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = total - TP - FP - FN

        ACC = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        SEN = TP / (TP + FN + 1e-8)
        SPEC = TN / (TN + FP + 1e-8)
        PPV = TP / (TP + FP + 1e-8)

        line = (
            f"{class_name},{TP},{TN},{FP},{FN},"
            f"{ACC:.6f},{SEN:.6f},{SPEC:.6f},{PPV:.6f}"
        )

        lines.append(line)

    return "\n".join(lines)


def evaluate_model(model, X_test, y_test):
    print("\n================ ĐÁNH GIÁ TEST SET ================")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss    : {test_loss:.6f}")
    print(f"Test Accuracy: {test_acc:.6f}")

    y_prob = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    print("\n================ CLASSIFICATION REPORT ================")
    print(report)

    with open(RESULT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    np.savetxt(RESULT_DIR / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    metrics_text = calculate_metrics_from_cm(cm)

    with open(RESULT_DIR / "metrics_ACC_SEN_SPEC_PPV.csv", "w", encoding="utf-8") as f:
        f.write(metrics_text)

    print("\n================ ACC / SEN / SPEC / PPV ================")
    print(metrics_text)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    print("[OK] Đã lưu confusion_matrix.png")
    print("[OK] Đã lưu classification_report.txt")
    print("[OK] Đã lưu metrics_ACC_SEN_SPEC_PPV.csv")


def save_model_and_weights(model):
    model.save(MODEL_DIR / "mina_mini_inceptionnet.h5")

    weights_dict = {}

    for layer in model.layers:
        layer_weights = layer.get_weights()

        if len(layer_weights) > 0:
            for idx, arr in enumerate(layer_weights):
                weights_dict[f"{layer.name}_{idx}"] = arr

    np.savez(MODEL_DIR / "weights_biases.npz", **weights_dict)

    print("[OK] Đã lưu models/mina_mini_inceptionnet.h5")
    print("[OK] Đã lưu models/weights_biases.npz")


# =========================================================
# 10. MAIN PROGRAM
# =========================================================

def main():
    print("===================================================")
    print(f"PROJECT: {PROJECT_NAME}")
    print("Chế độ dữ liệu: LOCAL DATASET")
    print("Không tải dữ liệu từ web.")
    print("===================================================")

    X_train, X_val, X_test, y_train, y_val, y_test = load_or_create_dataset()

    print("\n================ DATA SHAPE ================")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val  :", X_val.shape)
    print("y_val  :", y_val.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    model = build_mini_inceptionnet()

    print("\n================ MODEL SUMMARY ================")
    model.summary()

    total_params = model.count_params()

    print("\nTổng số parameters:", total_params)
    print("Mục tiêu main.pdf : 6,457 parameters")

    if total_params == 6457:
        print("[OK] Số parameter khớp main.pdf.")
    else:
        print("[WARNING] Số parameter chưa khớp 6,457. Cần kiểm tra lại kiến trúc.")

    print_compute_cost()

    history = train_model(model, X_train, y_train, X_val, y_val)

    best_model_path = MODEL_DIR / "best_mina_mini_inceptionnet.keras"
    best_model = tf.keras.models.load_model(best_model_path)

    plot_training_curve(history)
    evaluate_model(best_model, X_test, y_test)
    save_model_and_weights(best_model)

    print("\n[HOÀN TẤT] Huấn luyện Mini InceptionNet ECG xong.")


if __name__ == "__main__":
    main()