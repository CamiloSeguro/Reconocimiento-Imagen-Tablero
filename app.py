# app.py — Vision Insight Lab (GPT-4o Multimodal)
import os
import base64
import streamlit as st
from openai import OpenAI
from io import BytesIO

# ───────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Insight Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Vision Insight Lab")
st.caption("Analiza imágenes con **GPT-4o** — interpreta, describe y responde preguntas contextuales.")

# Estilo minimalista
st.markdown("""
<style>
div[data-testid="stFileUploader"] section div {
    text-align: center;
}
[data-testid="stImage"] img {
    border-radius: 12px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
}
textarea, input[type="text"] {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# UTILIDADES
# ───────────────────────────────────────────────────────────────
def encode_image(file) -> str:
    """Convierte la imagen subida a base64 para enviar al modelo."""
    return base64.b64encode(file.getvalue()).decode("utf-8")

def get_client(api_key: str) -> OpenAI:
    """Devuelve cliente OpenAI seguro si hay clave."""
    return OpenAI(api_key=api_key) if api_key else None

def analyze_image(base64_img: str, prompt: str, model: str = "gpt-4o") -> str:
    """Envía la imagen + prompt al modelo y obtiene streaming del análisis."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
            ],
        }
    ]

    client = get_client(st.session_state.api_key)
    full = ""
    message_box = st.empty()
    with client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1200,
        stream=True
    ) as stream:
        for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                full += delta.content
                message_box.markdown(full + "▌")
        message_box.markdown(full)
    return full

# ───────────────────────────────────────────────────────────────
# SIDEBAR — Configuración
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    api_input = st.text_input("🔑 OpenAI API key", type="password", placeholder="sk-...")
    if api_input:
        st.session_state.api_key = api_input.strip()
    elif "OPENAI_API_KEY" in st.secrets:
        st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.session_state.api_key = None

    model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o"], index=1)
    lang = st.selectbox("Idioma de respuesta", ["Español", "Inglés"], index=0)
    creativity = st.slider("Creatividad (temperature)", 0.0, 1.2, 0.4, step=0.1)
    show_prompts = st.toggle("Mostrar prompts creativos", value=False)

    if show_prompts:
        st.markdown("""
        - 🧩 **Describe** el contenido y estilo visual  
        - 🎨 **Analiza colores o composición**  
        - 🔍 **Identifica objetos o emociones**  
        - 🧠 **Imagina una historia** inspirada en la imagen  
        """)

# ───────────────────────────────────────────────────────────────
# SUBIDA DE IMAGEN
# ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📸 Sube una imagen para analizar",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    with st.expander("Vista previa", expanded=True):
        st.image(uploaded, use_container_width=True, caption=uploaded.name)

# Contexto opcional
st.markdown("#### 🗒️ Contexto adicional (opcional)")
user_context = st.text_area("¿Qué quieres que el modelo tenga en cuenta?", placeholder="Ejemplo: describe la atmósfera emocional y la posible historia detrás...")

# Selección rápida de modo
mode = st.radio(
    "Selecciona tipo de análisis:",
    ["Descripción general", "Análisis artístico", "Análisis técnico", "Historia creativa"],
    horizontal=True
)

prompt_map = {
    "Descripción general": "Describe con detalle lo que se observa en la imagen: objetos, colores, iluminación, contexto y emociones visuales.",
    "Análisis artístico": "Analiza la composición, uso de color, balance visual, estilo artístico y posibles influencias estéticas.",
    "Análisis técnico": "Evalúa calidad técnica: foco, exposición, profundidad de campo, iluminación, texturas y realismo.",
    "Historia creativa": "Imagina una historia corta inspirada en la escena de la imagen. Escríbela con un tono poético o narrativo."
}

# Botón de análisis
analyze = st.button("🔍 Analizar imagen", use_container_width=True, type="primary")

# ───────────────────────────────────────────────────────────────
# LÓGICA PRINCIPAL
# ───────────────────────────────────────────────────────────────
if analyze:
    if not uploaded:
        st.warning("📎 Por favor sube una imagen antes de analizar.")
    elif not st.session_state.api_key:
        st.warning("🔑 Ingresa tu API key en la barra lateral.")
    else:
        try:
            with st.spinner("Analizando imagen con GPT-4o..."):
                base64_img = encode_image(uploaded)
                base_prompt = prompt_map[mode]
                lang_suffix = "Responde en español." if lang == "Español" else "Respond in English."
                full_prompt = f"{base_prompt}\n\n{lang_suffix}"
                if user_context:
                    full_prompt += f"\n\nContexto adicional del usuario:\n{user_context}"
                result = analyze_image(base64_img, full_prompt, model=model)
                st.success("✅ Análisis completado")
                st.markdown("### 🧩 Resultado")
                st.write(result)
        except Exception as e:
            st.error(f"❌ Ocurrió un error: {e}")

# ───────────────────────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Desarrollado con ❤️ por Camilo Seguro — usando Streamlit + OpenAI GPT-4o")
