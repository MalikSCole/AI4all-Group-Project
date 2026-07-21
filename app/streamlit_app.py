"""Streamlit deployment for the Nutrition5k calorie classifier — Week 10 deliverable.

Run:  streamlit run app/streamlit_app.py

Loads the packaged checkpoint produced by the final cells of `Seeded Model.ipynb` and
classifies an uploaded food photo as Low / Medium / High calorie. If the weights are missing,
the app explains exactly how to export them (see also EXPORT_WEIGHTS.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.ordinal_cnn import CLASS_NAMES, CalorieClassifier  # noqa: E402

CHECKPOINT = REPO_ROOT / "models" / "final_ordinal_multitask_model.pth"
SCALER = REPO_ROOT / "models" / "regression_target_scaler.pkl"

st.set_page_config(page_title="Food Calorie Classifier", page_icon="🍽️", layout="centered")


@st.cache_resource
def load_model() -> CalorieClassifier | None:
    """Cached so the checkpoint is read once, not on every interaction."""
    try:
        return CalorieClassifier.load(CHECKPOINT, SCALER)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.session_state["model_load_error"] = str(exc)
        return None


def render_missing_weights() -> None:
    st.error("Model could not be loaded — the app cannot make predictions.")
    error = st.session_state.get("model_load_error")
    if error:
        st.code(error)
    st.markdown(
        f"""
The trained weights are **not in this repository**: `Seeded Model.ipynb` saves them to
`/kaggle/working/`, which Kaggle wipes when the session ends.

**To fix:** open `Seeded Model.ipynb` on Kaggle, use **Save & Run All (Commit)** — running
interactively and closing the tab does not persist the output — then download from the
completed version's Output tab:

- `final_ordinal_multitask_model.pth`
- `regression_target_scaler.pkl`

Place both in `{CHECKPOINT.parent.relative_to(REPO_ROOT)}/`. Full procedure: `EXPORT_WEIGHTS.md`.
"""
    )


def render_honest_caveats(metadata: dict) -> None:
    test_acc = metadata.get("test_accuracy")
    reported_accuracy = f"{test_acc:.1%}" if isinstance(test_acc, (int, float)) else "not available"
    with st.expander("How much should you trust this? (read before demoing)", expanded=False):
        st.markdown(
            f"""
**Reported test accuracy: {reported_accuracy}** against a 33.3% random baseline on three balanced
classes. That is a real result — roughly 2.2x baseline.

**Treat it as optimistic rather than a clean held-out estimate.** Test accuracy appears
several times across our notebooks (66.9%, 70.0%, 71.3%, 71.5%, 72.3%) before this model's
74.1% — and once you have checked the test set repeatedly and kept the changes that improved
it, the test number partly measures the search rather than generalization.

**What was verified clean:** `dish-level-data-leakage-check.ipynb` confirms 0 `dish_id`
overlap between train/val/test — the split is sound at the dish level.

**Where it will be least accurate:** Nutrition5k was captured in a controlled setting with a
fixed overhead camera and a specific culinary range. Phone photos at an angle, different
lighting, and cuisines the dataset under-represents all fall outside what the model saw. Food
datasets skew Western by default, so the people this serves worst are those whose food it
never saw.

**This is a class project, not a dietary tool.** Do not make food decisions from it.
"""
        )


def main() -> None:
    st.title("🍽️ Food Calorie Classifier")
    st.caption("Nutrition5k · ordinal multi-task CNN · AI4ALL")

    classifier = load_model()
    if classifier is None:
        render_missing_weights()
        st.stop()

    render_honest_caveats(classifier.metadata)

    uploaded = st.file_uploader(
        "Upload a food photo",
        type=["png", "jpg", "jpeg"],
        help="Works best on an overhead shot of a single plate — that is what it was trained on.",
    )

    if uploaded is None:
        st.info("Upload an image to classify it as Low, Medium, or High calorie.")
        return

    image = Image.open(uploaded)
    st.image(image, caption="Input", use_container_width=True)

    with st.spinner("Classifying..."):
        prediction = classifier.predict(image)

    st.subheader(f"Prediction: {prediction.class_name} calorie")

    if not prediction.is_consistent:
        # [0, 1] — "not above threshold 1, but above threshold 2" is incoherent for ordered
        # classes. Surfacing it is more honest than printing a confident label.
        st.warning(
            "The two ordinal thresholds disagree with each other, so this prediction is "
            "internally inconsistent. Treat it as the model being confused rather than as a "
            "real answer."
        )

    p1, p2 = prediction.threshold_probabilities
    st.markdown("**Threshold probabilities**")
    st.markdown(
        f"- P(above Low) = `{p1:.3f}` {'✅' if p1 >= 0.5 else '❌'}\n"
        f"- P(above Medium) = `{p2:.3f}` {'✅' if p2 >= 0.5 else '❌'}"
    )
    st.caption(
        f"Class = number of thresholds passed. {CLASS_NAMES[0]}=[0,0], "
        f"{CLASS_NAMES[1]}=[1,0], {CLASS_NAMES[2]}=[1,1]."
    )

    with st.expander("Where is the model looking? (Grad-CAM)", expanded=False):
        with st.spinner("Computing gradient heatmaps..."):
            from src.models.gradcam import explain

            explanations = explain(classifier.model, image)
        cols = st.columns(len(explanations))
        for col, (label, heat_img) in zip(cols, explanations):
            col.image(heat_img, caption=label, use_container_width=True)
        st.caption(
            "Grad-CAM: gradient-weighted evidence for each ordinal threshold, over the last "
            "conv block. Bright regions pushed that threshold's logit up. Read it as a "
            "diagnostic, not proof of reasoning — heat on the plate is what you want to see; "
            "heat on the table edge or shadows means the model is reading the scene, not "
            "the food."
        )

    if prediction.nutrition is not None:
        st.markdown("**Auxiliary regression estimates**")
        cols = st.columns(len(prediction.nutrition))
        units = {"calories": "kcal", "mass": "g", "fat": "g", "carb": "g", "protein": "g"}
        for col, (name, value) in zip(cols, prediction.nutrition.items()):
            col.metric(name.capitalize(), f"{value:.0f} {units.get(name, '')}")
        st.caption(
            "These come from the auxiliary head, which exists to give the shared trunk a richer "
            "training signal. They were never tuned for accuracy — read them as a rough sanity "
            "check, not as nutrition facts."
        )
    else:
        st.info(
            "Scaler not loaded, so the regression head's output is in standardized units and "
            "would be meaningless as nutrition numbers. Add `regression_target_scaler.pkl` to "
            "`models/` to see them."
        )


if __name__ == "__main__":
    main()
