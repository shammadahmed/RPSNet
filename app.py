import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Rock Paper Scissors",
    page_icon="✊",
    layout="wide",
)

# Must match train_ds.class_names order from Keras image_dataset_from_directory
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

LOW_CONFIDENCE_THRESHOLD = 0.60


@st.cache_resource
def load_model():
    # Automatically pick the newest available model file in the repo
    candidates = [
        "rock_paper_scissors_new_3.keras",
        "rock_paper_scissors.keras",
        "best_rps_model.keras",
    ]
    for path in candidates:
        if Path(path).exists():
            return tf.keras.models.load_model(path)
    raise FileNotFoundError(
        f"No model file found. Expected one of: {', '.join(candidates)}"
    )


model = load_model()


def get_model_target_size(loaded_model):
    """Read (width, height) dynamically from the model's input shape."""
    shape = getattr(loaded_model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if shape and len(shape) >= 3 and shape[1] and shape[2]:
        return (int(shape[2]), int(shape[1]))  # PIL expects (width, height)
    return (150, 150)


def predict_move(uploaded_file):
    # 1. Open image and apply EXIF orientation transpose for mobile photos
    raw_image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(raw_image).convert("RGB")

    # 2. Center-crop to 1:1 square and resize with bilinear interpolation
    #    (avoids squishing fingers on tall 9:16 or wide 16:9 photos)
    target_size = get_model_target_size(model)
    cropped_preview = ImageOps.fit(
        image,
        (400, 400),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )
    model_input_img = ImageOps.fit(
        image,
        target_size,
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )

    image_array = np.asarray(model_input_img, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    index = int(np.argmax(probabilities))

    prob_dict = {
        name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities)
    }

    return (
        CLASS_NAMES[index],
        float(probabilities[index]),
        cropped_preview,
        prob_dict,
    )


def get_result(move1, move2):
    if move1 == move2:
        return "It's a draw! 🤝"
    if WINNING_MOVES[move1] == move2:
        return "Player 1 wins! 🏆"
    return "Player 2 wins! 🏆"


def render_player_input(player_label, key_prefix):
    st.subheader(player_label)
    upload_tab, camera_tab = st.tabs(["📁 Upload Image", "📷 Use Camera"])
    with upload_tab:
        uploaded = st.file_uploader(
            f"Upload {player_label}'s hand",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"{key_prefix}_upload",
        )
    with camera_tab:
        camera_photo = st.camera_input(
            f"Take a photo for {player_label}",
            key=f"{key_prefix}_camera",
        )
    return camera_photo if camera_photo is not None else uploaded


def render_player_result(player_name, move, confidence, preview_img, prob_dict):
    st.image(preview_img, caption=f"{player_name} (Center-Cropped Input)", use_container_width=True)
    st.markdown(f"### {EMOJI[move]} {move.title()}")
    st.metric("Prediction confidence", f"{confidence:.1%}")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "Low confidence prediction. Try moving your hand closer to the center "
            "of the frame with a simpler background and even lighting."
        )

    with st.expander("See all class probabilities"):
        for cls_name in CLASS_NAMES:
            pct = prob_dict[cls_name]
            st.write(f"{EMOJI[cls_name]} **{cls_name.title()}**: `{pct:.1%}`")
            st.progress(min(max(pct, 0.0), 1.0))


st.title("✊ Rock Paper Scissors ✋")
st.write(
    "Upload or snap a hand image for each player. "
    "Our CNN predicts both moves and decides the winner."
)
st.info(
    "Tip: Fill most of the frame with a single hand gesture (rock, paper, or scissors) "
    "against a clear background."
)

col1, col2 = st.columns(2)
with col1:
    file1 = render_player_input("🔵 Player 1", "player1")
with col2:
    file2 = render_player_input("🔴 Player 2", "player2")

if file1 is not None and file2 is not None:
    if st.button("🎮 Play Round", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing both hand images..."):
                move1, confidence1, image1, probs1 = predict_move(file1)
                move2, confidence2, image2, probs2 = predict_move(file2)

            st.divider()
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                render_player_result("Player 1", move1, confidence1, image1, probs1)

            with result_col2:
                render_player_result("Player 2", move2, confidence2, image2, probs2)

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
    st.caption("Provide both Player 1 and Player 2 images to start playing.")