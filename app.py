import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ======================================================================
# Config
# ======================================================================
IMG_SIZE = 48
MODEL_PATH = "gtsrb_model.keras"
CONFIDENCE_THRESHOLD = 0.60  # below this, show an "uncertain" warning

CLASS_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End of no passing by vehicles over 3.5 tons",
}

# Sign category -> (label, emoji, colour)
CATEGORY_STYLE = {
    "prohibitory": ("Prohibitory", "🚫", "#ef4444"),
    "warning": ("Warning", "⚠️", "#f59e0b"),
    "mandatory": ("Mandatory", "🔵", "#3b82f6"),
    "priority": ("Priority", "🔶", "#f97316"),
    "end": ("End of restriction", "✅", "#6b7280"),
}


def sign_category(class_id: int) -> str:
    if class_id in (6, 32, 41, 42):
        return "end"
    if class_id in (11, 12, 13, 14):
        return "priority"
    if class_id in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17):
        return "prohibitory"
    if 18 <= class_id <= 31:
        return "warning"
    return "mandatory"  # 33-40


# ======================================================================
# Page setup & styling (dark theme)
# ======================================================================
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}
    #MainMenu, footer {visibility: hidden;}

    .hero {
        padding: 2.2rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 55%, #be185d 100%);
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 40px rgba(124, 58, 237, 0.35);
    }
    .hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800; color: #ffffff;}
    .hero p {margin: .5rem 0 0 0; font-size: 1.05rem; opacity: .92; color: #ffffff;}

    .card {
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
        background: linear-gradient(145deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,.3);
    }
    .label {
        font-size: .75rem; letter-spacing: .08em; text-transform: uppercase;
        opacity: .65; margin-bottom: .3rem; font-weight: 600;
    }
    .pred-name {font-size: 1.7rem; font-weight: 800; line-height: 1.2; margin: .2rem 0 .6rem 0;}
    .badge {
        display: inline-block; padding: .25rem .7rem; border-radius: 999px;
        font-size: .8rem; font-weight: 600; color: #fff;
    }
    .conf-big {font-size: 2.4rem; font-weight: 800; margin: .6rem 0 .1rem 0;}

    .bar-row {margin: .65rem 0;}
    .bar-head {display: flex; justify-content: space-between; font-size: .9rem; margin-bottom: .25rem;}
    .bar-track {
        height: 9px; border-radius: 999px;
        background: rgba(255,255,255,.10); overflow: hidden;
    }
    .bar-fill {height: 100%; border-radius: 999px;}

    .stat {text-align: center; padding: .6rem 0;}
    .stat .num {font-size: 1.5rem; font-weight: 800;}
    .stat .txt {font-size: .75rem; opacity: .65; text-transform: uppercase; letter-spacing: .06em;}
</style>
""",
    unsafe_allow_html=True,
)


# ======================================================================
# Model & prediction
# ======================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image, top_k: int = 5):
    probs = model.predict(preprocess(image), verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(int(i), CLASS_NAMES.get(int(i), str(i)), float(probs[i])) for i in top_idx]


def confidence_color(conf: float) -> str:
    if conf >= 0.80:
        return "#4ade80"
    if conf >= 0.50:
        return "#fbbf24"
    return "#f87171"


def render_bars(results):
    html = ""
    for _, name, conf in results:
        color = confidence_color(conf)
        html += (
            f'<div class="bar-row">'
            f'<div class="bar-head"><span>{name}</span><b>{conf:.1%}</b></div>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{conf * 100:.1f}%; background:{color};"></div></div>'
            f"</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


# ======================================================================
# Sidebar
# ======================================================================
with st.sidebar:
    st.markdown("## 🚦 About the model")
    st.markdown(
        "A convolutional neural network trained on the **German Traffic Sign "
        "Recognition Benchmark (GTSRB)**, using a class-weighted loss to handle "
        "class imbalance."
    )

    c1, c2 = st.columns(2)
    c1.markdown('<div class="stat"><div class="num">43</div><div class="txt">Classes</div></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="stat"><div class="num">~98%</div><div class="txt">Test accuracy</div></div>',
                unsafe_allow_html=True)
    c1.markdown(f'<div class="stat"><div class="num">{IMG_SIZE}×{IMG_SIZE}</div><div class="txt">Input size</div></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="stat"><div class="num">CNN</div><div class="txt">Architecture</div></div>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💡 Tips for best results")
    st.markdown(
        "- Crop tightly around a single sign\n"
        "- Use a well-lit, in-focus photo\n"
        "- Avoid heavy angles or obstructions"
    )

    st.divider()
    show_top_k = st.slider("Predictions to show", min_value=1, max_value=5, value=3)


# ======================================================================
# Hero
# ======================================================================
st.markdown(
    """
<div class="hero">
    <h1>🚦 Traffic Sign Recognition</h1>
    <p>Upload or capture a photo of a traffic sign and let the neural network identify it in an instant.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ======================================================================
# Load model
# ======================================================================
try:
    model = load_model()
except Exception as e:
    st.error(
        f"Couldn't load the model file `{MODEL_PATH}`. "
        "Make sure it's in the same folder as this app."
    )
    with st.expander("Error details"):
        st.code(str(e))
    st.stop()

# ======================================================================
# Main tabs
# ======================================================================
tab_predict, tab_classes = st.tabs(["🔍 Classify a sign", "📚 Supported signs"])

with tab_predict:
    source = st.radio(
        "Image source",
        ["Upload an image", "Use camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "Upload an image":
        file = st.file_uploader(
            "Drag and drop a traffic sign image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )
    else:
        file = st.camera_input("Take a picture of a traffic sign")

    if file is None:
        st.info("👆 Provide an image to get started.")
    else:
        image = Image.open(file)

        with st.spinner("Analysing image..."):
            results = predict(model, image, top_k=5)

        top_id, top_name, top_conf = results[0]
        cat_label, cat_emoji, cat_color = CATEGORY_STYLE[sign_category(top_id)]

        left, right = st.columns([1, 1.2], gap="large")

        with left:
            st.markdown('<div class="label">Your image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with right:
            st.markdown(
                f"""
<div class="card">
    <div class="label">Predicted sign</div>
    <div class="pred-name">{top_name}</div>
    <span class="badge" style="background:{cat_color};">{cat_emoji} {cat_label}</span>
    <div class="conf-big" style="color:{confidence_color(top_conf)};">{top_conf:.1%}</div>
    <div class="label">Confidence</div>
</div>
""",
                unsafe_allow_html=True,
            )

            if top_conf < CONFIDENCE_THRESHOLD:
                st.warning(
                    "The model is not very confident about this prediction. "
                    "Try a clearer, closer crop of the sign."
                )

            st.markdown('<div class="label">Top predictions</div>', unsafe_allow_html=True)
            render_bars(results[:show_top_k])

with tab_classes:
    st.markdown("The model recognises the following 43 sign types, grouped by category.")
    groups = {key: [] for key in CATEGORY_STYLE}
    for cid, name in CLASS_NAMES.items():
        groups[sign_category(cid)].append(name)

    cols = st.columns(2)
    for i, (key, names) in enumerate(groups.items()):
        label, emoji, _ = CATEGORY_STYLE[key]
        with cols[i % 2]:
            with st.expander(f"{emoji} {label} ({len(names)})"):
                for n in names:
                    st.markdown(f"- {n}")

# ======================================================================
# Footer
# ======================================================================
st.divider()
st.caption(
    "⚠️ Trained on German road signs (GTSRB). Signs from other countries may look "
    "different and be misclassified. This is a portfolio / demo project and should "
    "not be used for real driving decisions."
)
