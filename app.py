import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Rock Paper Scissors",
    page_icon="✊",
    layout="wide"
)

# These labels must match the order used during training.
CLASS_NAMES = ["paper", "rock", "scissors"]

WINNING_MOVES = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}

EMOJI = {
    "rock": "✊",
    "paper": "✋",
    "scissors": "✌️",
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "rock_paper_scissors.keras"
    )

model = load_model()


def predict_move(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.copy()

    # Match the model's training image dimensions.
    image = image.resize((150, 150))
    image_array = np.array(image, dtype=np.float32)

    # The model already has Rescaling(1./255).
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(
        image_array, verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    return (
        CLASS_NAMES[predicted_index],
        float(probabilities[predicted_index]),
        display_image,
    )


def get_result(move1, move2):
    if move1 == move2:
        return "It's a draw! 🤝"

    if WINNING_MOVES[move1] == move2:
        return "Player 1 wins! 🏆"

    return "Player 2 wins! 🏆"


st.title("✊ Rock Paper Scissors ✋")
st.write(
    "Upload a hand image for each player. "
    "Our CNN predicts both moves and decides the winner."
)

st.info(
    "Show one clear rock, paper, or scissors hand gesture "
    "in each image."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Player 1")
    file1 = st.file_uploader(
        "Upload Player 1's hand",
        type=["jpg", "jpeg", "png"],
        key="player1",
    )

with col2:
    st.subheader("🔴 Player 2")
    file2 = st.file_uploader(
        "Upload Player 2's hand",
        type=["jpg", "jpeg", "png"],
        key="player2",
    )

if file1 is not None and file2 is not None:
    if st.button("🎮 Play Round", type="primary",
                 use_container_width=True):

        try:
            with st.spinner("Analyzing both hand images..."):
                move1, confidence1, image1 = predict_move(file1)
                move2, confidence2, image2 = predict_move(file2)

            st.divider()
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.image(image1, caption="Player 1")
                st.markdown(
                    f"### {EMOJI[move1]} {move1.title()}"
                )
                st.metric(
                    "Prediction confidence",
                    f"{confidence1:.1%}"
                )

            with result_col2:
                st.image(image2, caption="Player 2")
                st.markdown(
                    f"### {EMOJI[move2]} {move2.title()}"
                )
                st.metric(
                    "Prediction confidence",
                    f"{confidence2:.1%}"
                )

            st.divider()
            st.subheader("🏆 Game Result")
            result = get_result(move1, move2)

            if move1 == move2:
                st.warning(result)
            else:
                st.success(result)
                st.balloons()

        except Exception as error:
            st.error(f"Could not process the images: {error}")

else:
    st.caption("Upload both images to start playing.")