"""Streamlit deployment for the Nutrition5k calorie-range classifier.

Run:
    streamlit run app/streamlit_app.py

The app loads the trained model checkpoint and classifies an uploaded food
photo into one of three broad categories:

- Low calorie range
- Medium calorie range
- High calorie range

This application is an educational AI demonstration and should not be used
as a medical, dietary, or exact nutrition-tracking tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.ordinal_cnn import CLASS_NAMES, CalorieClassifier  # noqa: E402

CHECKPOINT = REPO_ROOT / "models" / "final_ordinal_multitask_model.pth"
SCALER = REPO_ROOT / "models" / "regression_target_scaler.pkl"

st.set_page_config(
    page_title="Food Calorie Range Estimator",
    page_icon="🍽️",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model() -> CalorieClassifier | None:
    """Load and cache the trained model so it is not reloaded after each action."""
    try:
        return CalorieClassifier.load(CHECKPOINT, SCALER)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.session_state["model_load_error"] = str(exc)
        return None


# ---------------------------------------------------------------------------
# Error states
# ---------------------------------------------------------------------------

def render_missing_weights() -> None:
    """Display a user-friendly message when the model files are unavailable."""
    st.error(
        "The prediction model is currently unavailable, so the app cannot "
        "analyze food photos right now."
    )

    st.write(
        "The application loaded successfully, but the trained model files "
        "could not be found. The deployment needs to be updated before "
        "predictions can be made."
    )

    error = st.session_state.get("model_load_error")

    with st.expander("Developer setup information"):
        if error:
            st.markdown("**Model loading error**")
            st.code(error)

        st.markdown(
            f"""
The following files are required:

- `final_ordinal_multitask_model.pth`
- `regression_target_scaler.pkl`

Place both files inside:

`{CHECKPOINT.parent.relative_to(REPO_ROOT)}/`

The notebook saves these files to Kaggle's temporary `/kaggle/working/`
directory. To preserve them:

1. Open `Seeded Model.ipynb` in Kaggle.
2. Select **Save & Run All (Commit)**.
3. Wait for the committed notebook version to finish.
4. Open the completed version's **Output** tab.
5. Download both model files.
6. Add them to the project's `models/` directory.

See `EXPORT_WEIGHTS.md` for the complete export procedure.
"""
        )


# ---------------------------------------------------------------------------
# Page introduction
# ---------------------------------------------------------------------------

def render_intro() -> None:
    st.title("🍽️ Food Calorie Range Estimator")

    st.write(
        "Upload a photo of a meal and the AI model will estimate whether it "
        "belongs to a **low**, **medium**, or **high** calorie range."
    )

    st.info(
        "This app predicts a broad category. It does not calculate the exact "
        "number of calories in a meal."
    )

    st.caption(
        "Educational AI demonstration created using the Nutrition5k dataset."
    )


# ---------------------------------------------------------------------------
# Photo instructions
# ---------------------------------------------------------------------------

def render_photo_guidance() -> None:
    st.markdown("### Upload your meal photo")

    st.write(
        "For the best result, use a clear picture showing one complete plate "
        "of food."
    )

    with st.expander("Tips for taking a better photo"):
        st.markdown(
            """
- Show the entire plate.
- Take the photo from above when possible.
- Use bright, even lighting.
- Avoid heavy shadows.
- Try to include only one meal.
- Keep hands, packaging, and utensils from blocking the food.
- Avoid a busy or cluttered background.
"""
        )


# ---------------------------------------------------------------------------
# Reliability and limitations
# ---------------------------------------------------------------------------

def render_model_information(metadata: dict) -> None:
    """Explain model reliability using language suitable for general users."""
    test_acc = metadata.get("test_accuracy")

    reported_accuracy = (
        f"{test_acc:.1%}"
        if isinstance(test_acc, (int, float))
        else "not available"
    )

    with st.expander("How reliable is this app?"):
        st.markdown(
            f"""
During project testing, the model achieved approximately
**{reported_accuracy} accuracy** when choosing between three calorie-range
categories.

That means it often selects the correct category, but it will still make
mistakes. Its performance may also be lower on photos that look different
from the images used during training.

The model is most likely to struggle when:

- the photo is taken from the side;
- the image is dark or blurry;
- multiple plates are visible;
- part of the food is hidden;
- the background is visually distracting;
- the meal or cuisine was uncommon in the training data.

The model was primarily trained using controlled, overhead food photographs.
Everyday phone photos may produce less reliable results.
"""
        )

        st.warning(
            "Do not use this application for dieting, medical treatment, "
            "diabetes management, allergy decisions, or exact nutrition tracking."
        )

    with st.expander("Technical evaluation notes"):
        st.markdown(
            f"""
- **Reported test accuracy:** {reported_accuracy}
- **Number of categories:** 3
- **Random baseline:** approximately 33.3%
- **Dataset:** Nutrition5k
- **Task:** Ordered calorie-range classification
- **Classes:** Low, Medium, and High

The dataset split was checked to confirm that the same `dish_id` did not
appear in the training, validation, and test sets.

However, several model versions were evaluated against the test data during
development. Because the test set influenced some model-selection decisions,
the final reported accuracy may be somewhat optimistic.

The app should therefore be treated as a project demonstration rather than
a fully independent measurement of real-world performance.
"""
        )


# ---------------------------------------------------------------------------
# Prediction display
# ---------------------------------------------------------------------------

def render_prediction_result(prediction) -> None:
    """Display the selected calorie category in clear, nontechnical language."""
    st.markdown("### Estimated calorie range")

    result_messages = {
        "Low": (
            "The model estimates that this meal belongs to the lower calorie "
            "category compared with meals in its training data."
        ),
        "Medium": (
            "The model estimates that this meal belongs to the middle calorie "
            "category compared with meals in its training data."
        ),
        "High": (
            "The model estimates that this meal belongs to the higher calorie "
            "category compared with meals in its training data."
        ),
    }

    class_name = prediction.class_name

    st.success(f"Estimated category: **{class_name}**")

    st.write(
        result_messages.get(
            class_name,
            "The model placed this image into one of its three calorie-range "
            "categories.",
        )
    )

    st.caption(
        "The result is based on visual similarities between this image and "
        "food photos used to train the model. It is not an exact calorie count."
    )

    if not prediction.is_consistent:
        st.warning(
            "The model produced conflicting signals for this image, so the "
            "result may be unreliable. Try another photo with clearer lighting, "
            "less background clutter, and the full plate visible."
        )


# ---------------------------------------------------------------------------
# Internal model signals
# ---------------------------------------------------------------------------

def render_prediction_details(prediction) -> None:
    """Show ordinal threshold outputs without presenting them as exact confidence."""
    p1, p2 = prediction.threshold_probabilities

    with st.expander("See prediction details"):
        st.write(
            "The model makes its category decision in two steps. It first asks "
            "whether the image appears to be above the low range, and then asks "
            "whether it appears to be above the medium range."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Signal above Low",
                value=f"{p1:.0%}",
            )

        with col2:
            st.metric(
                label="Signal above Medium",
                value=f"{p2:.0%}",
            )

        st.markdown(
            """
The final category is selected using these two internal signals:

- Neither threshold passed → Low
- Only the first threshold passed → Medium
- Both thresholds passed → High
"""
        )

        st.caption(
            "These values are internal model signals. They should not be "
            "interpreted as guaranteed probabilities that the meal contains "
            "a particular number of calories."
        )

        if not prediction.is_consistent:
            st.warning(
                "The threshold signals do not follow the expected order. This "
                "indicates that the model is uncertain or confused by the image."
            )


# ---------------------------------------------------------------------------
# Grad-CAM explanation
# ---------------------------------------------------------------------------

def render_visual_explanation(classifier: CalorieClassifier, image: Image.Image) -> None:
    """Show which image regions influenced the model's internal decisions."""
    with st.expander("See what influenced the prediction"):
        st.write(
            "The images below highlight areas that had a stronger influence "
            "on the model's decision."
        )

        with st.spinner("Creating explanation images..."):
            from src.models.gradcam import explain

            explanations = explain(classifier.model, image)

        cols = st.columns(len(explanations))

        for col, (label, heat_img) in zip(cols, explanations):
            friendly_label = (
                label.replace("_", " ")
                .replace("threshold", "decision")
                .title()
            )

            col.image(
                heat_img,
                caption=friendly_label,
                use_container_width=True,
            )

        st.caption(
            "Brighter areas had more influence on the model's output. Ideally, "
            "the highlighted regions should appear on the food. Strong attention "
            "on shadows, the table, plate edges, or the background may indicate "
            "that the model is relying on irrelevant visual details."
        )

        st.caption(
            "Technical note: These visualizations use Grad-CAM. They are useful "
            "for inspecting the model, but they do not prove that the model "
            "understood the meal in the same way a person would."
        )


# ---------------------------------------------------------------------------
# Experimental nutrition output
# ---------------------------------------------------------------------------

def render_nutrition_estimates(prediction) -> None:
    """Display auxiliary regression outputs with strong limitations."""
    if prediction.nutrition is None:
        with st.expander("Experimental nutrition estimates"):
            st.info(
                "The optional nutrition-estimate component is not currently "
                "available because its supporting scaler file was not loaded."
            )

            st.caption(
                "Add `regression_target_scaler.pkl` to the `models/` directory "
                "to display these experimental values."
            )

        return

    with st.expander("Experimental nutrition estimates"):
        st.warning(
            "These values are experimental and may be substantially inaccurate. "
            "Do not treat them as nutrition facts or use them for health, dietary, "
            "or medical decisions."
        )

        st.write(
            "The model also produces rough estimates for several nutrition "
            "values. These estimates were included mainly to support the model's "
            "training and were not optimized as the primary output of the app."
        )

        units = {
            "calories": "kcal",
            "mass": "g",
            "fat": "g",
            "carb": "g",
            "protein": "g",
        }

        display_names = {
            "calories": "Calories",
            "mass": "Estimated weight",
            "fat": "Fat",
            "carb": "Carbohydrates",
            "protein": "Protein",
        }

        nutrition_items = list(prediction.nutrition.items())
        cols = st.columns(len(nutrition_items))

        for col, (name, value) in zip(cols, nutrition_items):
            col.metric(
                label=display_names.get(name, name.capitalize()),
                value=f"{value:.0f} {units.get(name, '')}".strip(),
            )

        st.caption(
            "These estimates should only be viewed as a rough model diagnostic. "
            "A food photo alone may not reveal ingredients, serving weight, oils, "
            "sauces, preparation methods, or hidden components."
        )


# ---------------------------------------------------------------------------
# About section
# ---------------------------------------------------------------------------

def render_about_section() -> None:
    with st.expander("About this project"):
        st.markdown(
            """
This application uses a convolutional neural network trained with food images
from the Nutrition5k dataset.

Instead of attempting to calculate an exact calorie count, the model places
each image into one of three ordered groups:

- Low calorie range
- Medium calorie range
- High calorie range

The categories are ordered, meaning that Medium sits between Low and High.
The model was designed to account for that relationship rather than treating
the three categories as completely unrelated labels.

The project also includes:

- auxiliary nutrition prediction tasks;
- model consistency checks;
- visual explanation heatmaps;
- dish-level dataset split validation;
- evaluation against a balanced three-class baseline.
"""
        )

        st.markdown("#### Important limitation")

        st.write(
            "A food image cannot reveal every ingredient or preparation detail. "
            "Two meals that look similar may have very different calorie totals "
            "because of portion size, oils, sauces, fillings, or cooking methods."
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    render_intro()

    classifier = load_model()

    if classifier is None:
        render_missing_weights()
        st.stop()

    render_photo_guidance()

    uploaded = st.file_uploader(
        "Choose a food photo",
        type=["png", "jpg", "jpeg"],
        help=(
            "Supported formats: PNG, JPG, and JPEG. The model works best with "
            "a clear overhead image of one complete plate."
        ),
    )

    if uploaded is None:
        st.info(
            "Upload a JPG or PNG image above to receive a Low, Medium, or High "
            "calorie-range prediction."
        )

        render_model_information(classifier.metadata)
        render_about_section()
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error(
            "The uploaded file could not be opened as an image. Please choose "
            "a valid PNG, JPG, or JPEG file."
        )
        return

    st.markdown("### Your uploaded photo")
    st.image(
        image,
        caption="Meal submitted for analysis",
        use_container_width=True,
    )

    with st.spinner("Analyzing the meal photo..."):
        try:
            prediction = classifier.predict(image)
        except (ValueError, RuntimeError) as exc:
            st.error(
                "The model could not analyze this image. Try uploading a "
                "different photo with clearer lighting and the full plate visible."
            )

            with st.expander("Technical error information"):
                st.code(str(exc))

            return

    render_prediction_result(prediction)
    render_prediction_details(prediction)
    render_visual_explanation(classifier, image)
    render_nutrition_estimates(prediction)
    render_model_information(classifier.metadata)
    render_about_section()

    st.divider()

    st.caption(
        "This application is an educational project. Its results are not "
        "medical advice and should not replace verified nutrition information."
    )


if __name__ == "__main__":
    main()
