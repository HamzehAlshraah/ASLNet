import os
import json
import zipfile
import threading
import mediapipe as mp
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from gtts import gTTS
import onnxruntime as ort

# =========================================
# Page config
# =========================================
st.set_page_config(
    page_title="ASL Sign Meaning App",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# Custom CSS
# =========================================
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(34,197,94,0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(59,130,246,0.10), transparent 25%),
        linear-gradient(180deg, #07111f 0%, #0b1220 100%);
    color: #f8fafc;}

.block-container {
    max-width: 1200px;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1324 0%, #101827 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin-bottom: 1.5rem;
}

.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 1.2rem;
    box-shadow: 0 14px 35px rgba(0,0,0,0.28);
    backdrop-filter: blur(10px);
}

.prediction-pill {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    border-radius: 20px;
    padding: 1.1rem 1rem;
    text-align: center;
    color: white;
    box-shadow: 0 12px 30px rgba(34,197,94,0.25);
}

.prediction-value {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.25rem;
}

.soft-divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 1rem 0;
}

.small-note {
    color: #cbd5e1;
    font-size: 0.92rem;
}

.lang-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
MODEL_ARCHIVE = os.path.join(BASE_DIR, "aslnet_v3.zip")
IMAGE_DIR = os.path.join(BASE_DIR, "Images")
IMG_SIZE = (224, 224)

class_names = ["BestWish", "ThankYou", "a", "am", "are", "bad", "bye", "e",
    "fine", "good", "hello", "how", "i", "is", "o", "the",
    "u", "what", "when", "who", "you", "your"]

gesture_info = {
    "BestWish": {"display": "Best Wish", "meaning_ar": "أطيب الأمنيات", "meaning_en": "Best wishes", "speak_text_en": "Best wishes", "speak_text_ar": "أطيب الأمنيات"},
    "ThankYou": {"display": "Thank You", "meaning_ar": "شكراً", "meaning_en": "Thank you", "speak_text_en": "Thank you", "speak_text_ar": "شكرا"},
    "a": {"display": "A", "meaning_ar": "الحرف A", "meaning_en": "Letter A", "speak_text_en": "Letter A", "speak_text_ar": "الحرف ايه"},
    "am": {"display": "Am", "meaning_ar": "أكون", "meaning_en": "Am", "speak_text_en": "Am", "speak_text_ar": "أكون"},
    "are": {"display": "Are", "meaning_ar": "تكون", "meaning_en": "Are", "speak_text_en": "Are", "speak_text_ar": "تكون"},
    "bad": {"display": "Bad", "meaning_ar": "سيء", "meaning_en": "Bad", "speak_text_en": "Bad", "speak_text_ar": "سيء"},
    "bye": {"display": "Bye", "meaning_ar": "مع السلامة", "meaning_en": "Bye", "speak_text_en": "Goodbye", "speak_text_ar": "مع السلامة"},
    "e": {"display": "E", "meaning_ar": "الحرف E", "meaning_en": "Letter E", "speak_text_en": "Letter E", "speak_text_ar": "الحرف إي"},
    "fine": {"display": "Fine", "meaning_ar": "بخير", "meaning_en": "Fine", "speak_text_en": "Fine", "speak_text_ar": "بخير"},
    "good": {"display": "Good", "meaning_ar": "جيد", "meaning_en": "Good", "speak_text_en": "Good", "speak_text_ar": "جيد"},
    "hello": {"display": "Hello", "meaning_ar": "مرحبا", "meaning_en": "Hello", "speak_text_en": "Hello", "speak_text_ar": "مرحبا"},
    "how": {"display": "How", "meaning_ar": "كيف", "meaning_en": "How", "speak_text_en": "How", "speak_text_ar": "كيف"},
    "i": {"display": "I", "meaning_ar": "أنا", "meaning_en": "I", "speak_text_en": "I", "speak_text_ar": "أنا"},
    "is": {"display": "Is", "meaning_ar": "يكون", "meaning_en": "Is", "speak_text_en": "Is", "speak_text_ar": "يكون"},
    "o": {"display": "O", "meaning_ar": "الحرف O", "meaning_en": "Letter O", "speak_text_en": "Letter O", "speak_text_ar": "الحرف أو"},
    "the": {"display": "The", "meaning_ar": "أل التعريف", "meaning_en": "The", "speak_text_en": "The", "speak_text_ar": "أل التعريف"},
    "u": {"display": "U", "meaning_ar": "الحرف U", "meaning_en": "Letter U", "speak_text_en": "Letter U", "speak_text_ar": "الحرف يو"},
    "what": {"display": "What", "meaning_ar": "ماذا", "meaning_en": "What", "speak_text_en": "What", "speak_text_ar": "ماذا"},
    "when": {"display": "When", "meaning_ar": "متى", "meaning_en": "When", "speak_text_en": "When", "speak_text_ar": "متى"},
    "who": {"display": "Who", "meaning_ar": "من", "meaning_en": "Who", "speak_text_en": "Who", "speak_text_ar": "من"},
    "you": {"display": "You", "meaning_ar": "أنت", "meaning_en": "You", "speak_text_en": "You", "speak_text_ar": "أنت"},
    "your": {"display": "Your", "meaning_ar": "لك", "meaning_en": "Your", "speak_text_en": "Your", "speak_text_ar": "لك"}
}
gesture_image_paths = {name: os.path.join(IMAGE_DIR, f"{name}.jpg")
                       for name in class_names}
@st.cache_resource
def load_asl_model():
    if os.path.isfile(MODEL_PATH):
        with open(os.path.join(BASE_DIR, "labels.json"), encoding="utf-8") as f:
            labels = json.load(f)
        model_source = MODEL_PATH
    else:
        with zipfile.ZipFile(MODEL_ARCHIVE) as archive:
            model_source = archive.read("model.onnx")
            labels = json.loads(archive.read("labels.json"))
    session = ort.InferenceSession(model_source, providers=["CPUExecutionProvider"])
    return session, labels

model, class_names = load_asl_model()
@st.cache_resource
def load_hand_detector():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=os.path.join(BASE_DIR, "hand_landmarker.task")
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options), threading.Lock()


def crop_detected_hands(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    detector, lock = load_hand_detector()
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.asarray(image, dtype=np.uint8).copy(),
    )
    with lock:
        result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None, image

    width, height = image.size
    points = [(p.x * width, p.y * height)
              for hand in result.hand_landmarks for p in hand]
    xs, ys = zip(*points)
    # Keep both hands together and leave room around fingers and wrist.
    margin = max(12, int(max(max(xs) - min(xs), max(ys) - min(ys)) * 0.20))
    box = (
        max(0, int(np.floor(min(xs))) - margin),
        max(0, int(np.floor(min(ys))) - margin),
        min(width, int(np.ceil(max(xs))) + margin),
        min(height, int(np.ceil(max(ys))) + margin),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, image
    cropped = image.crop(box)
    annotated = image.copy()
    ImageDraw.Draw(annotated).rectangle(
        (box[0], box[1], box[2] - 1, box[3] - 1),
        outline=(34, 197, 94), width=max(2, min(width, height) // 150),
    )
    return cropped, annotated


def preprocess_pil_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr / 255.0 - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None].astype(np.float32)
    return arr

def predict_image(image: Image.Image):
    x = preprocess_pil_image(image)
    logits = model.run(None, {model.get_inputs()[0].name: x})[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    preds = exp_logits / np.sum(exp_logits)
    pred_idx = int(np.argmax(preds))
    pred_conf = float(preds[pred_idx])
    raw_name = class_names[pred_idx]
    info = gesture_info.get(raw_name,{"display": raw_name,"meaning_ar": raw_name,
        "meaning_en": raw_name,"speak_text_en": raw_name,"speak_text_ar": raw_name})

    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = []
    for i in top3_idx:
        cls = class_names[int(i)]
        inf = gesture_info.get(cls,{"display": cls,"meaning_ar": cls,"meaning_en": cls,
            "speak_text_en": cls,"speak_text_ar": cls})
        top3.append((cls, inf["display"], float(preds[int(i)])))

    return raw_name, info, pred_conf, top3
def text_to_speech_bytes(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer
def image_exists(path):
    return os.path.exists(path) and os.path.isfile(path)
for i in range (2):
    st.write("")
language = st.radio("Choose Language / اختر اللغة",["English", "العربية"],horizontal=True)

if language == "العربية":
    page_title = "تطبيق معنى إشارات اليد"
    page_subtitle = "صوّر الإشارة أو ارفع صورة، وسيعرض التطبيق الاسم والمعنى وخيار تشغيل الصوت."
    input_method_label = "طريقة الإدخال"
    input_options = ["التصوير", "رفع صورة"]
    capture_title = "التصوير من الكاميرا"
    upload_title = "رفع صورة"
    camera_label = "التقط صورة"
    uploader_label = "اختر صورة"
    notes_title = "ملاحظات"
    notes_1 = "حاول تكون اليد واضحة وفي منتصف الصورة."
    notes_2 = "إذا كانت الإضاءة ضعيفة أو الخلفية مزعجة، قد تقل الدقة."
    notes_3 = "يمكنك استخدام التصوير المباشر أو رفع صورة من الجهاز."
    sidebar_title = "معلومات المشروع"
    sidebar_desc = "هذا التطبيق يتعرف على إشارة اليد من صورة واحدة، ثم يعرض المعنى ويشغّل الصوت."
    how_to_use = "طريقة الاستخدام"
    supported_signs = "الإشارات المدعومة"
    result_title = "النتيجة"
    captured_image_title = "الصورة"
    predicted_sign_title = "الإشارة المتوقعة"
    meaning_ar_title = "المعنى بالعربي"
    meaning_en_title = "المعنى بالإنجليزي"
    example_sign_title = "صورة الإشارة"
    top3_title = "أعلى 3 احتمالات"
    play_en_audio = "🔊 تشغيل الصوت بالإنجليزي"
    play_ar_audio = "🔊 تشغيل الصوت بالعربي"
    no_input_message = "صوّر صورة أو ارفع صورة حتى تظهر النتيجة."
else:
    page_title = "ASL Sign Meaning App"
    page_subtitle = "Capture a sign or upload an image to get the predicted class, meaning, and audio."
    input_method_label = "Input Method"
    input_options = ["Camera", "Upload Image"]
    capture_title = "Capture from Camera"
    upload_title = "Upload Image"
    camera_label = "Take a photo"
    uploader_label = "Choose an image"
    notes_title = "Notes"
    notes_1 = "Try to keep the hand clear and centered."
    notes_2 = "Poor lighting or busy backgrounds may reduce accuracy."
    notes_3 = "You can use either live camera capture or upload an image from your device."
    sidebar_title = "Project Info"
    sidebar_desc = "This app predicts a hand sign from a single image, then shows the meaning and plays audio."
    how_to_use = "How to use"
    supported_signs = "Supported Signs"
    result_title = "Prediction Result"
    captured_image_title = "Captured Image"
    predicted_sign_title = "Predicted Sign"
    meaning_ar_title = "Meaning (Arabic)"
    meaning_en_title = "Meaning (English)"
    example_sign_title = "Example Sign Image"
    top3_title = "Top 3 Predictions"
    play_en_audio = "🔊 Play English Audio"
    play_ar_audio = "🔊 Play Arabic Audio"
    no_input_message = "Capture or upload an image first to see the result."
with st.sidebar:
    st.markdown(f"## 🧠 {sidebar_title}")
    st.write(sidebar_desc)
    st.write("**Model:** MobileNetV2")
    st.write("**Classes:** 22")
    st.write("**Input Size:** 224 x 224")
    st.markdown(f"### 📘 {how_to_use}")
    if language == "العربية":
        st.markdown("""
        - اختر التصوير أو رفع صورة
        - التقط الإشارة أو ارفعها
        - شاهد اسم الإشارة
        - اقرأ معناها
        - شغّل الصوت
        """)
    else:
        st.markdown("""- Choose camera or image upload
        - Capture the sign or upload it
        - View the predicted sign
        - Read the meaning
        - Play the audio""")
    st.markdown(f"### 🏷️ {supported_signs}")
    with st.expander("Show sign list" if language == "English" else "عرض قائمة الإشارات", expanded=False):
        for cls in class_names:
            info = gesture_info[cls]
            img_path = gesture_image_paths.get(cls, "")
            c1, c2 = st.columns([1, 3])
            with c1:
                if image_exists(img_path):
                    st.image(img_path, width=42)
                else:
                    st.markdown("🖼️")
            with c2:
                st.markdown(f"**{info['display']}**")
                st.caption(info["meaning_en"] if language == "English" else info["meaning_ar"])
st.markdown(f'<div class="hero-title">{page_title}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-subtitle">{page_subtitle}</div>', unsafe_allow_html=True)
input_method = st.radio(input_method_label,input_options,horizontal=True)
uploaded_or_captured_image = None
left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if (language == "English" and input_method == "Camera") or (language == "العربية" and input_method == "التصوير"):
        st.subheader(f"📷 {capture_title}")
        uploaded_or_captured_image = st.camera_input(camera_label)
    else:
        st.subheader(f"🖼️ {upload_title}")
        uploaded_or_captured_image = st.file_uploader(uploader_label,type=["jpg", "jpeg", "png"])

    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(f"ℹ️ {notes_title}")
    st.write(notes_1)
    st.write(notes_2)
    st.write(notes_3)
    st.markdown('</div>', unsafe_allow_html=True)
if uploaded_or_captured_image is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded_or_captured_image)).convert("RGB")
    hand_image, annotated_image = crop_detected_hands(image)
    if hand_image is None:
        st.image(image, use_container_width=True)
        st.warning(
            "لم يتم العثور على يد واضحة. أظهر اليد كاملة مع إضاءة جيدة ثم حاول مجددًا."
            if language == "العربية" else
            "No clear hand detected. Show the whole hand in good lighting and try again."
        )
        st.stop()
    raw_name, info, conf, top3 = predict_image(hand_image)

    st.markdown(f"## {result_title}")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(f"🖼️ {captured_image_title}")
        st.image(annotated_image, use_container_width=True)
        st.image(
            hand_image,
            caption="الصورة المرسلة للموديل" if language == "العربية" else "Image sent to the model",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(f"🤖 {predicted_sign_title}")

        st.markdown(f"""<div class="prediction-pill">
                <div>{predicted_sign_title}</div>
                <div class="prediction-value">{info["display"]}</div>
                <div>Confidence: {conf:.2%}</div>
            </div>""",unsafe_allow_html=True)

        st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

        st.write(f"**{meaning_ar_title}:** {info['meaning_ar']}")
        st.write(f"**{meaning_en_title}:** {info['meaning_en']}")

        predicted_img_path = gesture_image_paths.get(raw_name, "")
        if image_exists(predicted_img_path):
            st.markdown(f"### {example_sign_title}")
            st.image(predicted_img_path, width=180)

        st.markdown(f"### {top3_title}")
        for _, display_name, score in top3:
            st.progress(max(0.0, min(score, 1.0)), text=f"{display_name} — {score:.2%}")

        st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

        col_audio1, col_audio2 = st.columns(2)

        with col_audio1:
            if st.button(play_en_audio):
                audio_bytes = text_to_speech_bytes(info["speak_text_en"], lang="en")
                st.audio(audio_bytes, format="audio/mp3")

        with col_audio2:
            if st.button(play_ar_audio):
                audio_bytes = text_to_speech_bytes(info["speak_text_ar"], lang="ar")
                st.audio(audio_bytes, format="audio/mp3")

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info(no_input_message)
