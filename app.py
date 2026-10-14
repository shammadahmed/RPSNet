```python
import random
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Rock Paper Scissors",
    page_icon="✊",
    layout="centered",
)

MODEL_PATH = "rock_paper_scissors.keras"
IMAGE_SIZE = (150, 150)

# Must match the class order used during model training.
CLASS_NAMES = ["paper", "rock", "scissors"]

EMOJI = {
    "rock": "✊",
    "paper": "✋",
    "scissors": "✌️",
}

WINNING_MOVES = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}


# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # The model already contains Rescaling(1./255).
    # Do not normalize the image a second time.
    image_array = np.asarray(image, dtype=np.float32)

    return np.expand_dims(image_array, axis=0)


# --------------------------------------------------
# Predict hand gesture
# --------------------------------------------------
def predict_move(image: Image.Image, model):
    processed_image = preprocess_image(image)

    probabilities = model.predict(
        processed_image,
        verbose=0,
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "Model output count does not match CLASS_NAMES."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_move = CLASS_NAMES[predicted_index]

    return predicted_move, probabilities


# --------------------------------------------------
# Determine winner
# --------------------------------------------------
def decide_winner(player_move: str, computer_move: str) -> str:
    if player_move == computer_move:
        return "draw"

    if WINNING_MOVES[player_move] == computer_move:
        return "player"

    return "computer"


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------
st.title("✊ Rock Paper Scissors")
st.write("Upload a hand image and play against the computer!")

with st.expander("Tips for better predictions"):
    st.markdown(
        """
        - Keep your hand fully visible and centered.
        - Use a clear image with good lighting.
        - Avoid cluttered backgrounds where possible.
        - The model may struggle with photos that differ
          significantly from its training dataset.
        """
    )

# Load model safely.
try:
    model = load_model()
except Exception as exc:
    st.error(
        f"Could not load `{MODEL_PATH}`. "
        "Ensure the model file exists in your repository."
    )

    with st.expander("Technical details"):
        st.code(str(exc))

    st.stop()


# Upload image.
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        st.error("Could not open this image. Please try another.")
        st.stop()

    st.image(
        image,
        caption="Your uploaded hand image",
        use_container_width=True,
    )

    if st.button(
        "Play Round",
        type="primary",
        use_container_width=True,
    ):
        try:
            player_move, probabilities = predict_move(
                image,
                model,
            )

            computer_move = random.choice(CLASS_NAMES)

            outcome = decide_winner(
                player_move,
                computer_move,
            )

            st.session_state["rps_result"] = {
                "player_move": player_move,
                "computer_move": computer_move,
                "probabilities": probabilities.tolist(),
                "outcome": outcome,
            }

        except Exception as exc:
            st.error("Prediction failed. Please try another image.")

            with st.expander("Technical details"):
                st.code(str(exc))


# --------------------------------------------------
# Display round result
# --------------------------------------------------
if "rps_result" in st.session_state:
    result = st.session_state["rps_result"]

    player_move = result["player_move"]
    computer_move = result["computer_move"]
    probabilities = result["probabilities"]
    outcome = result["outcome"]

    st.divider()
    st.subheader("Round Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Your Move",
            f"{EMOJI[player_move]} {player_move.title()}",
        )

    with col2:
        st.metric(
            "Computer's Move",
            f"{EMOJI[computer_move]} {computer_move.title()}",
        )

    if outcome == "player":
        st.success("🎉 You win!")

    elif outcome == "computer":
        st.error("🤖 The computer wins!")

    else:
        st.info("🤝 It's a draw!")

    # Display model scores for debugging.
    st.subheader("Prediction Confidence")

    st.caption(
        "These scores show the model's output, not a guarantee "
        "that its prediction is correct."
    )

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities,
    ):
        st.write(
            f"{EMOJI[class_name]} "
            f"{class_name.title()}: {probability:.2%}"
        )

        st.progress(
            float(np.clip(probability, 0.0, 1.0))
        )

    if st.button("Clear Round", use_container_width=True):
        del st.session_state["rps_result"]
        st.rerun()

else:
    st.info("Upload an image and click Play Round to begin.")

st.divider()
st.caption("Built with Streamlit and TensorFlow.")
```