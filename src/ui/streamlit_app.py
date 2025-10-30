"""
Streamlit prototype for LexiMind (summarization, emotion, topic).
Run from repo root: streamlit run streamlit_app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Stable absolute import; ensure repo root is on PYTHONPATH (running from repo root is standard)
try:
    from ..api.inference import load_models, summarize_text, classify_emotion, topic_for_text
except Exception as e:
    st.error(f"Failed to import inference helpers: {e}")
    raise

st.set_page_config(page_title="LexiMind demo", layout="wide")

MODEL_CONFIG = {
    "checkpoint_path": "checkpoints/best.pt",  # change to your trained checkpoint
    "tokenizer_path": "artifacts/tokenizer.json",  # JSON produced by TextPreprocessor.save_tokenizer
    "device": "cpu",
}
try:
    models = load_models(MODEL_CONFIG)
except Exception as exc:
    st.error(f"Failed to load models: {exc}")
    st.stop()

st.sidebar.title("LexiMind")
task = st.sidebar.selectbox("Task", ["Summarize", "Emotion", "Topic", "Search demo"])
compression = st.sidebar.slider("Compression (summary length)", 0.1, 1.0, 0.25)
show_attn = st.sidebar.checkbox("Show attention heatmap (collect_attn)", value=False)

st.sidebar.markdown("Demo controls")
sample_choice = st.sidebar.selectbox("Use sample text", ["None", "Gutenberg sample", "News sample"])

SAMPLES = {
    "Gutenberg sample": (
        "It was the best of times, it was the worst of times, it was the age of wisdom, "
        "it was the age of foolishness..."
    ),
    "News sample": (
        "Markets rallied today as tech stocks posted gains amid broad optimism over earnings..."
    ),
}

st.title("LexiMind — Summarization, Emotion, Topic (Prototype)")

if sample_choice != "None":
    input_text = st.text_area("Input text", value=SAMPLES[sample_choice], height=280)
else:
    input_text = st.text_area("Input text", value="", height=280)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Output")
    if st.button("Run"):
        if not input_text.strip():
            st.warning("Enter some text or select a sample to run the model.")
        else:
            if task == "Summarize":
                summary, attn_data = summarize_text(input_text, compression=compression, collect_attn=show_attn, models=models)
                st.markdown("**Summary**")
                st.write(summary)
                if show_attn and attn_data is not None:
                    st.markdown("**Attention heatmap (averaged heads)**")
                    src_tokens = attn_data.get("src_tokens", None)
                    tgt_tokens = attn_data.get("tgt_tokens", None)
                    weights = attn_data.get("weights", None)
                    if weights is not None:
                        arr = np.array(weights)
                        if arr.ndim == 4:
                            arr = arr.mean(axis=(0,1))
                        elif arr.ndim == 3:
                            arr = arr.mean(axis=0)
                        fig = ff.create_annotated_heatmap(
                            z=arr.tolist(),
                            x=src_tokens if src_tokens else [f"tok{i}" for i in range(arr.shape[1])],
                            y=tgt_tokens if tgt_tokens else [f"tok{i}" for i in range(arr.shape[0])],
                            colorscale="Viridis",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Attention data not available from the model.")
            elif task == "Emotion":
                probs, labels = classify_emotion(input_text, models=models)
                st.markdown("**Emotion predictions (multi-label probabilities)**")
                df = pd.DataFrame({"emotion": labels, "prob": probs})
                fig = px.bar(df, x="emotion", y="prob", color="prob", range_y=[0,1])
                st.plotly_chart(fig, use_container_width=True)
            elif task == "Topic":
                topic_id, topic_terms = topic_for_text(input_text, models=models)
                st.markdown("**Topic cluster**")
                st.write(f"Cluster ID: {topic_id}")
                st.write("Top terms:", ", ".join(topic_terms))
            elif task == "Search demo":
                st.info("Search demo will be available when ingestion is run (see scripts).")

with col2:
    st.subheader("Model & Info")
    st.markdown(f"*Model loaded:* {'yes' if models.get('loaded', False) else 'no'}")
    st.markdown(f"*Device:* {models.get('device', MODEL_CONFIG['device'])}")
    st.markdown("**Notes**")
    st.markdown("- Attention visualization depends on model support to return attention.")
    st.markdown("- For long inputs the UI truncates tokens for heatmap clarity.")