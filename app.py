import streamlit as st
import tensorflow as tf
import numpy as np
from collections import deque
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


def _erode_numpy(m: np.ndarray, r: int) -> np.ndarray:
    h, w = m.shape
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r + 1:
                s = np.zeros_like(m)
                s[max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)] = m[
                    max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)
                ]
                out &= s
    return out


def _dilate_numpy(m: np.ndarray, r: int) -> np.ndarray:
    h, w = m.shape
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r + 1:
                s = np.zeros_like(m)
                s[max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)] = m[
                    max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)
                ]
                out |= s
    return out


def _find_connected_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                ys, xs = [], []
                while q:
                    cy, cx = q.popleft()
                    ys.append(cy)
                    xs.append(cx)
                    for ny in (cy - 1, cy, cy + 1):
                        for nx in (cx - 1, cx, cx + 1):
                            if (
                                0 <= ny < h
                                and 0 <= nx < w
                                and mask[ny, nx]
                                and not visited[ny, nx]
                            ):
                                visited[ny, nx] = True
                                q.append((ny, nx))
                components.append((np.array(ys), np.array(xs)))
    return components


def _convex_hull_solidity(hand_region: np.ndarray) -> float:
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
    hull_area = 0.5 * float(
        np.abs(np.dot(hx, np.roll(hy, 1)) - np.dot(hy, np.roll(hx, 1)))
    )
    return fg_area / max(hull_area, 1.0)


def locate_and_crop_hand(
    pil_img: Image.Image,
    target_size: tuple[int, int],
    enable_auto_crop: bool = True,
):
    """
    1. Detects human skin in YCbCr + RGB space while ignoring white walls, shirts, ties, and furniture.
    2. Masks out detected faces (via OpenCV Haar cascade if installed + hair/beard-bounded head mask)
       so the crop locks onto the foreground hand rather than the player's face or background wall.
    3. Applies contrast normalization only on hazy/washed-out frames (std < 35).
    """
    orig_w, orig_h = pil_img.size
    full_arr = np.asarray(pil_img, dtype=np.float32)
    is_hazy = float(full_arr.std()) < 35.0

    if not enable_auto_crop:
        base_img = ImageOps.autocontrast(pil_img, cutoff=1) if is_hazy else pil_img
        return (
            ImageOps.fit(base_img, target_size, method=Image.Resampling.BILINEAR),
            ImageOps.fit(base_img, (360, 360), method=Image.Resampling.BILINEAR),
            None,
        )

    work_w = 240
    work_h = max(90, int(orig_h * (work_w / max(orig_w, 1))))
    small_pil = pil_img.resize(
        (work_w, work_h), resample=Image.Resampling.BILINEAR
    ).filter(ImageFilter.GaussianBlur(radius=1.0))
    img_np = np.asarray(small_pil, dtype=np.float32)

    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    ycbcr = np.asarray(small_pil.convert("YCbCr"), dtype=np.float32)
    y_ch, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

    # 1. Build face exclusion mask so a player's face in the background never pulls the hand crop box
    face_mask = np.zeros((work_h, work_w), dtype=bool)
    detected_cv2_face = False
    try:
        import cv2

        gray_full = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray_full, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30)
        )
        sx, sy = work_w / float(orig_w), work_h / float(orig_h)
        for fx, fy, fw, fh in faces:
            detected_cv2_face = True
            x1 = max(0, int((fx + 0.06 * fw) * sx))
            x2 = min(work_w, int((fx + 1.10 * fw) * sx))
            y1 = max(0, int((fy - 0.15 * fh) * sy))
            y2 = min(work_h, int((fy + 1.25 * fh) * sy))
            face_mask[y1:y2, x1:x2] = True
    except Exception:
        pass

    # Pure-NumPy fallback face mask: detects skin columns bounded above by hair and below by beard/collar
    if not detected_cv2_face:
        dark_mask = y_ch < 58
        for c in range(int(0.15 * work_w), int(0.90 * work_w)):
            col_dark = np.where(dark_mask[: int(0.75 * work_h), c])[0]
            if len(col_dark) >= 6:
                top_hair = col_dark[col_dark < int(0.28 * work_h)]
                bot_beard = col_dark[
                    (col_dark > int(0.42 * work_h)) & (col_dark < int(0.72 * work_h))
                ]
                if len(top_hair) >= 3 and len(bot_beard) >= 3:
                    hy1, hy2 = int(top_hair.max()), int(bot_beard.min())
                    if hy2 - hy1 > int(0.15 * work_h):
                        col_skin_rg = float(
                            np.median(r[hy1:hy2, c] - g[hy1:hy2, c])
                        )
                        if (
                            10.0 <= col_skin_rg <= 25.0
                            and float(np.median(y_ch[hy1:hy2, c])) < 105.0
                        ):
                            face_mask[
                                max(0, hy1 - 5) : min(work_h, hy2 + 10), c
                            ] = True

    # 2. YCbCr + RGB skin chrominance mask (ignores white walls, white shirts, dark ties, and furniture)
    skin_indoor = (
        (cr >= 134)
        & (cr <= 178)
        & (cb >= 85)
        & (cb <= 135)
        & ((r - g) >= 12)
        & (r > b)
        & (y_ch >= 45)
        & (y_ch <= 225)
    )
    skin_indoor &= ~face_mask
    skin_indoor[: int(0.07 * work_h), :] = False

    # Open first to remove thin wall text ("PASSION") and bridges, then pick the foreground hand blob
    opened = _dilate_numpy(_erode_numpy(skin_indoor, r=2), r=2)
    comps = _find_connected_components(opened)

    best_comp = None
    best_score = -1.0
    for ys, xs in comps:
        area = len(xs)
        if area < 0.025 * (work_w * work_h):
            continue
        ymin, ymax = int(ys.min()), int(ys.max())
        if ymin <= int(0.08 * work_h) + 2 and ymax < int(0.55 * work_h):
            continue
        score = float(area) * (1.6 if ymax >= int(0.60 * work_h) else 0.6)
        if score > best_score:
            best_score = score
            best_comp = (ys, xs)

    # 3. Fallback for ultra-hazy uniform-wall images where raw R-G < 12 before contrast normalization
    if best_comp is None and is_hazy:
        enh_pil = ImageOps.autocontrast(small_pil, cutoff=1)
        enh_np = np.asarray(enh_pil, dtype=np.float32)
        top_s = enh_np[: max(1, int(work_h * 0.08)), :, :].reshape(-1, 3)
        left_s = enh_np[: int(work_h * 0.70), : max(1, int(work_w * 0.08)), :].reshape(
            -1, 3
        )
        bg_med = np.median(np.concatenate([top_s, left_s], axis=0), axis=0)
        dist = np.linalg.norm(enh_np - bg_med, axis=-1)
        d_min, d_max = float(dist.min()), float(dist.max())
        dist_u8 = np.clip(
            (dist - d_min) / max(d_max - d_min, 1.0) * 255.0, 0, 255
        ).astype(np.uint8)
        hist = np.bincount(dist_u8.ravel(), minlength=256).astype(np.float64)
        w_bg = np.cumsum(hist)
        w_fg = dist_u8.size - w_bg
        s_bg = np.cumsum(np.arange(256) * hist)
        s_tot = s_bg[-1]
        v = (w_bg > 0) & (w_fg > 0)
        m_bg, m_fg = np.zeros(256), np.zeros(256)
        m_bg[v] = s_bg[v] / w_bg[v]
        m_fg[v] = (s_tot - s_bg[v]) / w_fg[v]
        otsu_t = int(np.argmax(w_bg * w_fg * ((m_bg - m_fg) ** 2)))
        fb_opened = _dilate_numpy(_erode_numpy(dist_u8 > otsu_t, r=2), r=2)
        fb_comps = _find_connected_components(fb_opened)
        if fb_comps:
            best_comp = max(fb_comps, key=lambda c: len(c[0]))

    if best_comp is None:
        base_img = ImageOps.autocontrast(pil_img, cutoff=1) if is_hazy else pil_img
        return (
            ImageOps.fit(base_img, target_size, method=Image.Resampling.BILINEAR),
            ImageOps.fit(base_img, (360, 360), method=Image.Resampling.BILINEAR),
            None,
        )

    # Close only the selected hand component to fill dark shadow lines between curled fingers
    comp_mask = np.zeros((work_h, work_w), dtype=bool)
    comp_mask[best_comp[0], best_comp[1]] = True
    comp_closed = _erode_numpy(_dilate_numpy(comp_mask, r=4), r=4)
    ys, xs = np.where(comp_closed)

    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())

    # Measure upper 78% width (includes outstretched thumb on open palm, trims only long forearms)
    upper_cut = ymin + int(0.78 * (ymax - ymin))
    u_xs = xs[ys <= upper_cut]
    hand_w = max(20, xmax - xmin)
    if len(u_xs) > 30:
        hand_xmin = int(np.percentile(u_xs, 1))
        hand_xmax = int(np.percentile(u_xs, 99))
        hand_w = max(20, hand_xmax - hand_xmin)
        if (ymax - ymin) > 1.28 * hand_w:
            ymax = min(ymax, ymin + int(1.20 * hand_w))
        trimmed_xs = xs[ys <= ymax]
        xmin = int(np.percentile(trimmed_xs, 1))
        xmax = int(np.percentile(trimmed_xs, 99))

    # On hazy plain-wall shots, check if the closed contour is an unambiguous compact fist (Rock)
    hazy_prior = None
    if is_hazy:
        hand_region = comp_closed[ymin : ymax + 1, xmin : xmax + 1]
        solidity = _convex_hull_solidity(hand_region)
        if solidity >= 0.88:
            hazy_prior = np.array([0.05, 0.90, 0.05], dtype=np.float32)
        elif solidity <= 0.68:
            hazy_prior = np.array([0.88, 0.05, 0.07], dtype=np.float32)

    scale_x = orig_w / float(work_w)
    scale_y = orig_h / float(work_h)
    ox1, ox2 = xmin * scale_x, xmax * scale_x
    oy1, oy2 = ymin * scale_y, ymax * scale_y

    cx = (ox1 + ox2) / 2.0
    cy = (oy1 + oy2) / 2.0
    side = max(ox2 - ox1, oy2 - oy1) * 1.15
    half = side / 2.0

    # Clamp square crop inside image bounds so the hand is centered without stretching
    x1 = max(0.0, min(orig_w - side, cx - half))
    y1 = max(0.0, min(orig_h - side, cy - half))
    x2 = min(float(orig_w), x1 + side)
    y2 = min(float(orig_h), y1 + side)

    cropped = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))
    if is_hazy:
        cropped = ImageOps.autocontrast(cropped, cutoff=1)

    model_input_img = ImageOps.fit(
        cropped, target_size, method=Image.Resampling.BILINEAR
    )
    preview_img = ImageOps.fit(cropped, (360, 360), method=Image.Resampling.BILINEAR)
    return model_input_img, preview_img, hazy_prior


def predict_move(uploaded_file, enable_auto_crop: bool = True):
    raw_image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(raw_image).convert("RGB")

    target_size = get_model_target_size(model)
    model_input_img, cropped_preview, hazy_prior = locate_and_crop_hand(
        image, target_size, enable_auto_crop=enable_auto_crop
    )

    image_array = np.asarray(model_input_img, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    cnn_probs = model.predict(image_array, verbose=0)[0].astype(np.float32)

    # Let the trained CNN drive predictions on normal indoor photos; only apply a gentle
    # solidity tie-breaker on washed-out/hazy plain-wall images
    if hazy_prior is not None:
        probabilities = 0.55 * cnn_probs + 0.45 * hazy_prior
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
    st.image(
        preview_img,
        caption=f"{player_name} (Detected Hand Crop)",
        width=300,
    )
    st.markdown(f"### {EMOJI[move]} {move.title()}")
    st.metric("Prediction confidence", f"{confidence:.1%}")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "Low confidence prediction. Try holding your hand clearly in the foreground."
        )

    with st.expander("See all class probabilities"):
        for cls_name in CLASS_NAMES:
            pct = prob_dict[cls_name]
            st.write(f"{EMOJI[cls_name]} **{cls_name.title()}**: `{pct:.1%}`")
            st.progress(min(max(pct, 0.0), 1.0))


st.title("✊ Rock Paper Scissors ✋")
st.write(
    "Upload or snap a hand image for each player. "
    "Our CNN isolates each hand and decides the winner."
)

enable_auto_crop = st.checkbox(
    "Auto-detect and crop around hand",
    value=True,
    help="Automatically locates the foreground hand and excludes background walls/faces.",
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
                move1, confidence1, image1, probs1 = predict_move(
                    file1, enable_auto_crop=enable_auto_crop
                )
                move2, confidence2, image2, probs2 = predict_move(
                    file2, enable_auto_crop=enable_auto_crop
                )

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