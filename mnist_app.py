"""
MNIST Single Perceptron — Streamlit App
Run with: streamlit run mnist_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import io

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Perceptron",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    h1 { color: #00d4ff; font-family: 'Courier New', monospace; }
    h2, h3 { color: #ffffff; }
    .stMetric { background: #1e2130; border-radius: 10px; padding: 10px; }
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #0077ff);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-size: 1rem; font-weight: bold;
    }
    .stButton>button:hover { opacity: 0.85; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 MNIST Single Perceptron Classifier")
st.markdown("Train a single-layer perceptron on MNIST with **80% train / 20% test** split, then predict your own handwritten digit.")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    epochs     = st.slider("Epochs",          5,  50, 30)
    batch_size = st.slider("Batch Size",     32, 256, 128, step=32)
    lr         = st.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)
    st.markdown("---")
    st.markdown("**Model Architecture**")
    st.code("Input (784)\n   ↓\nDense(10, softmax)\n   ↓\nOutput (0–9)", language="text")

# ─── Session State ────────────────────────────────────────────────────────────
if "model"   not in st.session_state: st.session_state.model   = None
if "history" not in st.session_state: st.session_state.history = None
if "X_test"  not in st.session_state: st.session_state.X_test  = None
if "y_test"  not in st.session_state: st.session_state.y_test  = None
if "y_pred"  not in st.session_state: st.session_state.y_pred  = None

# ─── Train Button ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 3])
with col1:
    train_btn = st.button("🚀 Train Model")

if train_btn:
    with st.spinner("Downloading MNIST & training..."):

        # Load & split
        (X_all, y_all), (X_extra, y_extra) = mnist.load_data()
        X_all = np.concatenate([X_all, X_extra])
        y_all = np.concatenate([y_all, y_extra])
        X_all = X_all.astype("float32") / 255.0

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
        )

        y_train_cat = to_categorical(y_train, 10)

        # Build model
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer=SGD(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Progress bar
        progress = st.progress(0, text="Training epoch 0 / " + str(epochs))
        history_loss, history_val_loss = [], []
        history_acc,  history_val_acc  = [], []

        class StreamlitCallback:
            def on_epoch_end(self, epoch, logs=None):
                pct = int((epoch + 1) / epochs * 100)
                progress.progress(pct, text=f"Training epoch {epoch+1} / {epochs}  —  acc: {logs['accuracy']:.3f}")
                history_loss.append(logs["loss"])
                history_val_loss.append(logs["val_loss"])
                history_acc.append(logs["accuracy"])
                history_val_acc.append(logs["val_accuracy"])

        from tensorflow.keras.callbacks import LambdaCallback
        cb = LambdaCallback(on_epoch_end=lambda e, logs: StreamlitCallback().on_epoch_end(e, logs))

        hist = model.fit(
            X_train, y_train_cat,
            epochs=epochs, batch_size=batch_size,
            validation_split=0.1, verbose=0, callbacks=[cb],
        )

        progress.empty()

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

        st.session_state.model   = model
        st.session_state.history = hist.history
        st.session_state.X_test  = X_test
        st.session_state.y_test  = y_test
        st.session_state.y_pred  = y_pred

    st.success("✅ Training complete!")

# ─── Results ──────────────────────────────────────────────────────────────────
if st.session_state.model is not None:
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    X_test = st.session_state.X_test
    history = st.session_state.history

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

    # Per-class report
    with st.expander("📋 Full Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        import pandas as pd
        df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.header("📉 Training Curves (Error)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0e1117")
    ep = range(1, len(history["loss"]) + 1)
    for ax in axes:
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")

    axes[0].plot(ep, history["loss"],     color="#ff4b4b", lw=2, label="Train Loss")
    axes[0].plot(ep, history["val_loss"], color="#ff4b4b", lw=2, ls="--", label="Val Loss")
    axes[0].set_title("Loss over Epochs", color="white")
    axes[0].set_xlabel("Epoch", color="white")
    axes[0].legend(facecolor="#1e2130", labelcolor="white")
    axes[0].grid(alpha=0.2)

    axes[1].plot(ep, history["accuracy"],     color="#00d4ff", lw=2, label="Train Acc")
    axes[1].plot(ep, history["val_accuracy"], color="#00d4ff", lw=2, ls="--", label="Val Acc")
    axes[1].set_title("Accuracy over Epochs", color="white")
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].legend(facecolor="#1e2130", labelcolor="white")
    axes[1].grid(alpha=0.2)

    st.pyplot(fig)

    st.markdown("---")
    st.header("🔢 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(9, 7), facecolor="#0e1117")
    ax2.set_facecolor("#1e2130")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10), ax=ax2, linewidths=0.3)
    ax2.set_title("Confusion Matrix", color="white", fontsize=13)
    ax2.set_xlabel("Predicted", color="white")
    ax2.set_ylabel("True", color="white")
    ax2.tick_params(colors="white")
    st.pyplot(fig2)

    st.markdown("---")
    st.header("❌ Misclassified Examples")

    wrong_idx = np.where(y_pred != y_test)[0]
    st.write(f"**{len(wrong_idx):,}** misclassified out of **{len(y_test):,}** — error rate: **{len(wrong_idx)/len(y_test)*100:.1f}%**")

    fig3, axes3 = plt.subplots(3, 6, figsize=(13, 6), facecolor="#0e1117")
    for i, ax in enumerate(axes3.flat):
        ax.set_facecolor("#0e1117")
        if i >= len(wrong_idx): ax.axis("off"); continue
        idx = wrong_idx[i]
        ax.imshow(X_test[idx], cmap="gray")
        ax.set_title(f"T:{y_test[idx]} P:{y_pred[idx]}", color="red", fontsize=8)
        ax.axis("off")
    st.pyplot(fig3)

    # ─── Predict Uploaded Image ───────────────────────────────────────────────
    st.markdown("---")
    st.header("🖼️ Predict Your Own Digit")
    st.markdown("Upload a **28×28 grayscale image** of a handwritten digit (or any image — it will be resized).")

    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("L").resize((28, 28))
        img_arr = np.array(img).astype("float32") / 255.0

        # Invert if background is white
        if img_arr.mean() > 0.5:
            img_arr = 1.0 - img_arr

        col_a, col_b = st.columns([1, 2])
        with col_a:
            fig4, ax4 = plt.subplots(facecolor="#0e1117")
            ax4.imshow(img_arr, cmap="gray")
            ax4.set_facecolor("#0e1117")
            ax4.axis("off")
            ax4.set_title("Your Image (28×28)", color="white")
            st.pyplot(fig4)

        with col_b:
            pred_probs = st.session_state.model.predict(img_arr.reshape(1, 28, 28), verbose=0)[0]
            pred_digit = np.argmax(pred_probs)
            confidence = pred_probs[pred_digit] * 100

            st.markdown(f"### Predicted Digit: **{pred_digit}**")
            st.markdown(f"Confidence: **{confidence:.1f}%**")

            fig5, ax5 = plt.subplots(figsize=(7, 3), facecolor="#0e1117")
            ax5.set_facecolor("#1e2130")
            bars = ax5.bar(range(10), pred_probs * 100,
                           color=["#00d4ff" if i == pred_digit else "#444" for i in range(10)])
            ax5.set_xticks(range(10))
            ax5.set_xlabel("Digit", color="white")
            ax5.set_ylabel("Confidence (%)", color="white")
            ax5.set_title("Prediction Probabilities", color="white")
            ax5.tick_params(colors="white")
            for spine in ax5.spines.values(): spine.set_edgecolor("#444")
            ax5.grid(alpha=0.2)
            st.pyplot(fig5)
