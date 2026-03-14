"""
MNIST Classification - Single Layer Perceptron
Using scikit-learn (no tensorflow) for Streamlit Cloud compatibility
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import struct, gzip, urllib.request, os

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="MNIST Perceptron", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    h1 { color: #00d4ff; font-family: 'Courier New', monospace; }
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #0077ff);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-size: 1rem; font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 MNIST Single Perceptron Classifier")
st.markdown("Train a single-layer perceptron on MNIST with **80% train / 20% test** split.")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    epochs = st.slider("Epochs", 5, 50, 30)
    lr     = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
    st.markdown("---")
    st.markdown("**Model Architecture**")
    st.code("Input (784)\n   ↓\nDense(10, softmax)\n   ↓\nOutput (0–9)", language="text")

# ─── Download MNIST ───────────────────────────────────────────────────────────
def download_mnist():
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }
    data = {}
    for key, fname in files.items():
        if not os.path.exists(fname):
            urllib.request.urlretrieve(base + fname, fname)
        with gzip.open(fname, "rb") as f:
            if "images" in key:
                _, n, r, c = struct.unpack(">IIII", f.read(16))
                data[key] = np.frombuffer(f.read(), np.uint8).reshape(n, r * c)
            else:
                f.read(8)
                data[key] = np.frombuffer(f.read(), np.uint8)
    X = np.concatenate([data["train_images"], data["test_images"]]).astype("float32") / 255.0
    y = np.concatenate([data["train_labels"], data["test_labels"]])
    return X, y

# ─── Session State ────────────────────────────────────────────────────────────
for key in ["model", "history", "X_test", "y_test", "y_pred", "scaler", "X_test_display"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Train ────────────────────────────────────────────────────────────────────
if st.button("🚀 Train Model"):
    with st.spinner("Downloading MNIST..."):
        X, y = download_mnist()
    st.success(f"✅ MNIST loaded — {len(X):,} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_test_display = X_test.copy()  # save unscaled for showing images

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    progress = st.progress(0, text=f"Training epoch 1 / {epochs}")
    val_acc_curve = []

    model = SGDClassifier(
        loss="log_loss", learning_rate="constant", eta0=lr,
        max_iter=1, warm_start=True, random_state=42, n_jobs=-1,
    )

    for epoch in range(epochs):
        model.max_iter += 1
        model.fit(X_train, y_train)
        train_acc = model.score(X_train[:5000], y_train[:5000])
        val_acc   = model.score(X_test[:2000],  y_test[:2000])
        val_acc_curve.append(val_acc)
        pct = int((epoch + 1) / epochs * 100)
        progress.progress(pct, text=f"Epoch {epoch+1}/{epochs} — train acc: {train_acc:.3f} | val acc: {val_acc:.3f}")

    progress.empty()
    y_pred = model.predict(X_test)

    st.session_state.model          = model
    st.session_state.history        = val_acc_curve
    st.session_state.X_test         = X_test
    st.session_state.X_test_display = X_test_display
    st.session_state.y_test         = y_test
    st.session_state.y_pred         = y_pred
    st.session_state.scaler         = scaler
    st.success("✅ Training complete!")

# ─── Results ──────────────────────────────────────────────────────────────────
if st.session_state.model is not None:
    y_test  = st.session_state.y_test
    y_pred  = st.session_state.y_pred
    X_test  = st.session_state.X_test
    history = st.session_state.history
    X_disp  = st.session_state.X_test_display

    st.markdown("---")
    st.header("📊 Metrics")

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Accuracy",  f"{acc*100:.2f}%")
    c2.metric("🔍 Precision", f"{prec*100:.2f}%")
    c3.metric("📡 Recall",    f"{rec*100:.2f}%")
    c4.metric("⚖️ F1-Score",  f"{f1*100:.2f}%")

    with st.expander("📋 Full Classification Report"):
        import pandas as pd
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.markdown("---")
    st.header("📉 Training Curve (Error)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(history)+1), history, color="#00d4ff", lw=2, label="Val Accuracy")
    ax.set_title("Accuracy over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.markdown("---")
    st.header("🔢 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10), ax=ax2)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    st.pyplot(fig2)

    st.markdown("---")
    st.header("❌ Misclassified Examples")
    wrong_idx = np.where(y_pred != y_test)[0]
    st.write(f"**{len(wrong_idx):,}** misclassified — error rate: **{len(wrong_idx)/len(y_test)*100:.1f}%**")

    fig3, axes3 = plt.subplots(3, 6, figsize=(13, 6))
    for i, ax in enumerate(axes3.flat):
        if i >= len(wrong_idx): ax.axis("off"); continue
        idx = wrong_idx[i]
        ax.imshow(X_disp[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"T:{y_test[idx]} P:{y_pred[idx]}", color="red", fontsize=8)
        ax.axis("off")
    st.pyplot(fig3)

    st.markdown("---")
    st.header("🖼️ Predict Your Own Digit")
    st.markdown("Upload a photo of a handwritten digit (0–9).")

    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img     = Image.open(uploaded).convert("L").resize((28, 28))
        img_arr = np.array(img).astype("float32") / 255.0
        if img_arr.mean() > 0.5:
            img_arr = 1.0 - img_arr

        col_a, col_b = st.columns([1, 2])
        with col_a:
            fig4, ax4 = plt.subplots()
            ax4.imshow(img_arr, cmap="gray")
            ax4.axis("off")
            ax4.set_title("Your Image (28×28)")
            st.pyplot(fig4)

        with col_b:
            img_scaled = st.session_state.scaler.transform(img_arr.reshape(1, -1))
            pred_digit = st.session_state.model.predict(img_scaled)[0]
            proba      = st.session_state.model.predict_proba(img_scaled)[0]
            confidence = proba[pred_digit] * 100

            st.markdown(f"### Predicted Digit: **{pred_digit}**")
            st.markdown(f"Confidence: **{confidence:.1f}%**")

            fig5, ax5 = plt.subplots(figsize=(7, 3))
            ax5.bar(range(10), proba * 100,
                    color=["#00d4ff" if i == pred_digit else "#cccccc" for i in range(10)])
            ax5.set_xticks(range(10))
            ax5.set_xlabel("Digit")
            ax5.set_ylabel("Confidence (%)")
            ax5.set_title("Prediction Probabilities")
            ax5.grid(alpha=0.2)
            st.pyplot(fig5)
