import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter

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
    candidates = [
        "rock_paper_scissors_new_3.keras",
        "best_rps_model.keras",
        "rock_paper_scissors.keras",
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
        return (int(shape[2]), int(shape[1]))
    return (150, 150)


def _convex_hull_solidity(hand_region: np.ndarray) -> float:
    """Compute 2D foreground solidity (foreground_area / convex_hull_area) in pure NumPy."""
    h_ys, h_xs = np.where(hand_region)
    if len(h_xs) < 80:
        return 1.0

    fg_area = float(len(h_xs))
    unique_rows = np.unique(h_ys)
    pts = []
    for r in unique_rows:
        row_xs = h_xs[h_ys == r]
        pts.append((float(row_xs.min()), float(r)))
        if row_xs.max() != row_xs.min():
            pts.append((float(row_xs.max()), float(r)))
    pts = sorted(set(pts))
    if len(pts) < 3:
        return 1.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 1.0

    hx = np.array([p[0] for p in hull])
    hy = np.array([p[1] for p in hull])
    hull_area = 0.5 * float(np.abs(np.dot(hx, np.roll(hy, 1)) - np.dot(hy, np.roll(hx, 1))))
    return fg_area / max(hull_area, 1.0)


def _count_finger_runs(hand_region: np.ndarray, hand_w: int) -> int:
    """Count distinct extended finger segments across slices in the upper half of the hand."""
    min_run = max(2, int(0.04 * hand_w))
    row_counts = []
    for frac in np.linspace(0.12, 0.48, 9):
        r = int(frac * hand_region.shape[0])
        if 0 <= r < hand_region.shape[0]:
            row = hand_region[r, :]
            padded = np.concatenate([[False], row, [False]])
            starts = np.where(padded[1:] & ~padded[:-1])[0]
            ends = np.where(~padded[1:] & padded[:-1])[0]
            widths = ends - starts
            row_counts.append(int(np.sum(widths >= min_run)))
    return int(np.percentile(row_counts, 75)) if row_counts else 1


def locate_and_crop_hand(pil_img: Image.Image, target_size: tuple[int, int]):
    """
    1. Applies ImageOps.autocontrast to fix foggy / low-contrast webcam photos.
    2. Segments the hand from the background wall (even when off-center on the left/right edge).
    3. Trims off the lower forearm so the 1:1 crop centers tightly on the hand/fist.
    4. Returns (model_input_img, preview_img, geom_probs).
    """
    enhanced = ImageOps.autocontrast(pil_img, cutoff=1)
    work_w = 240
    work_h = max(60, int(enhanced.height * (work_w / max(enhanced.width, 1))))
    small = enhanced.resize((work_w, work_h), resample=Image.Resampling.BILINEAR)
    small_blur = small.filter(ImageFilter.GaussianBlur(radius=2))
    arr = np.asarray(small_blur, dtype=np.float32)

    # Sample background from top and cleanest upper side border (avoiding bottom where arm enters)
    top_strip = arr[:max(1, int(work_h * 0.08)), :, :].reshape(-1, 3)
    left_strip = arr[:int(work_h * 0.70), :max(1, int(work_w * 0.08)), :].reshape(-1, 3)
    right_strip = arr[:int(work_h * 0.70), max(0, int(work_w * 0.92)):, :].reshape(-1, 3)

    top_med = np.median(top_strip, axis=0)
    left_med = np.median(left_strip, axis=0)
    right_med = np.median(right_strip, axis=0)
    if np.linalg.norm(left_med - top_med) <= np.linalg.norm(right_med - top_med):
        bg_pixels = np.concatenate([top_strip, left_strip], axis=0)
    else:
        bg_pixels = np.concatenate([top_strip, right_strip], axis=0)
    bg_color = np.median(bg_pixels, axis=0)

    diff = arr - bg_color
    dist = np.sqrt(
        0.30 * (diff[:, :, 0] ** 2)
        + 0.59 * (diff[:, :, 1] ** 2)
        + 0.11 * (diff[:, :, 2] ** 2)
        + np.linalg.norm(diff, axis=-1) ** 2
    )

    d_min, d_max = float(dist.min()), float(dist.max())
    if d_max - d_min < 5.0:
        return (
            ImageOps.fit(enhanced, target_size, method=Image.Resampling.BILINEAR),
            ImageOps.fit(enhanced, (400, 400), method=Image.Resampling.BILINEAR),
            None,
        )

    # Otsu threshold on background distance map
    dist_u8 = np.clip((dist - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(dist_u8.ravel(), minlength=256).astype(np.float64)
    total = dist_u8.size
    sum_total = np.dot(np.arange(256), hist)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(np.arange(256) * hist)
    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.zeros(256)
    mean_fg = np.zeros(256)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    var_between = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
    thresh = int(np.argmax(var_between))
    mask = dist_u8 > thresh

    ys, xs = np.where(mask)
    if len(xs) < 0.03 * total:
        return (
            ImageOps.fit(enhanced, target_size, method=Image.Resampling.BILINEAR),
            ImageOps.fit(enhanced, (400, 400), method=Image.Resampling.BILINEAR),
            None,
        )

    ymin, ymax = int(np.percentile(ys, 1)), int(np.percentile(ys, 99))
    upper_cut = ymin + int(0.65 * (ymax - ymin))
    upper_mask = mask.copy()
    upper_mask[upper_cut:, :] = False
    u_ys, u_xs = np.where(upper_mask)
    if len(u_xs) < 50:
        return (
            ImageOps.fit(enhanced, target_size, method=Image.Resampling.BILINEAR),
            ImageOps.fit(enhanced, (400, 400), method=Image.Resampling.BILINEAR),
            None,
        )

    hand_xmin = int(np.percentile(u_xs, 2))
    hand_xmax = int(np.percentile(u_xs, 98))
    hand_w = max(15, hand_xmax - hand_xmin)

    # Isolate upper hand/fist region (trimming lower forearm) for geometry analysis
    hand_ymax = min(work_h - 1, ymin + int(1.05 * hand_w))
    hand_region = mask[ymin : hand_ymax + 1, hand_xmin : hand_xmax + 1]

    solidity = _convex_hull_solidity(hand_region)
    max_fingers = _count_finger_runs(hand_region, hand_w)

    geom_probs = None
    # Class order: ["paper", "rock", "scissors"]
    if solidity >= 0.83 and max_fingers <= 1:
        # Compact closed fist with no separated finger projections -> Rock
        geom_probs = np.array([0.03, 0.92, 0.05], dtype=np.float32)
    elif solidity < 0.72 or max_fingers >= 3:
        # Wide splayed fingers with large inter-finger gaps -> Paper
        geom_probs = np.array([0.92, 0.03, 0.05], dtype=np.float32)
    elif max_fingers == 2 or (0.72 <= solidity < 0.83):
        # Two extended fingers -> Scissors
        geom_probs = np.array([0.08, 0.07, 0.85], dtype=np.float32)

    # Map tight hand bounding box back to original image resolution
    scale_x = enhanced.width / float(work_w)
    scale_y = enhanced.height / float(work_h)
    orig_xmin = hand_xmin * scale_x
    orig_xmax = hand_xmax * scale_x
    orig_ymin = ymin * scale_y
    orig_ymax = min(enhanced.height, (ymin + int(1.20 * hand_w)) * scale_y)

    cx = (orig_xmin + orig_xmax) / 2.0
    cy = (orig_ymin + orig_ymax) / 2.0
    side = max(orig_xmax - orig_xmin, orig_ymax - orig_ymin) * 1.35
    half = side / 2.0
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(enhanced.width, cx + half))
    y2 = int(min(enhanced.height, cy + half))

    cropped = enhanced.crop((x1, y1, x2, y2))
    model_input_img = ImageOps.fit(cropped, target_size, method=Image.Resampling.BILINEAR)
    preview_img = ImageOps.fit(cropped, (400, 400), method=Image.Resampling.BILINEAR)
    return model_input_img, preview_img, geom_probs


def predict_move(uploaded_file):
    # 1. Open image and apply EXIF orientation transpose for mobile photos
    raw_image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(raw_image).convert("RGB")

    # 2. Auto-crop around the hand (even if off-center), trim forearm, and normalize contrast
    target_size = get_model_target_size(model)
    model_input_img, cropped_preview, geom_probs = locate_and_crop_hand(image, target_size)

    image_array = np.asarray(model_input_img, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    cnn_probs = model.predict(image_array, verbose=0)[0].astype(np.float32)

    # 3. Fuse CNN probabilities with hand-contour geometry prior when background is clean
    if geom_probs is not None:
        probabilities = 0.40 * cnn_probs + 0.60 * geom_probs
        probabilities = probabilities / np.sum(probabilities)
    else:
        probabilities = cnn_probs

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
    st.image(preview_img, caption=f"{player_name} (Auto-Cropped & Contrast-Enhanced Hand)", width=320)
    st.markdown(f"### {EMOJI[move]} {move.title()}")
    st.metric("Prediction confidence", f"{confidence:.1%}")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "Low confidence prediction. Try holding your hand against a plain background "
            "with clear lighting."
        )

    with st.expander("See all class probabilities"):
        for cls_name in CLASS_NAMES:
            pct = prob_dict[cls_name]
            st.write(f"{EMOJI[cls_name]} **{cls_name.title()}**: `{pct:.1%}`")
            st.progress(min(max(pct, 0.0), 1.0))


st.title("✊ Rock Paper Scissors ✋")
st.write(
    "Upload or snap a hand image for each player. "
    "Our detector isolates each hand, enhances contrast, and predicts the winner."
)
st.info(
    "Show one clear rock, paper, or scissors hand gesture against a simple background."
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