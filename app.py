from pathlib import Path
import pandas as pd
import streamlit as st
import cv2
from inference import MODEL_PATHS, load_model, process_frame, png_bytes, image_from_bytes, zip_images, process_video

st.set_page_config(page_title="Baches Perú | Visión artificial", page_icon="🛣️", layout="wide", initial_sidebar_state="expanded")
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"], [data-testid="stApp"]{font-family:'DM Sans',sans-serif}
[data-testid="stApp"]{background:radial-gradient(ellipse at 70% 0%,#123047 0%,#09111f 48%);color:#eff8ff}
[data-testid="stSidebar"]{background:#0d192a;border-right:1px solid #24384d}
h1,h2,h3{letter-spacing:-.035em}h1{font-weight:800!important}
[data-testid="stMetric"]{background:#12243a;border:1px solid #25435c;padding:18px;border-radius:16px}
[data-testid="stMetricLabel"]{color:#a9c0d1}
div.stButton>button[kind="primary"],div.stDownloadButton>button[kind="primary"]{background:#29d17b;color:#082019;border:0;font-weight:800;border-radius:10px;min-height:44px}
div.stButton>button,div.stDownloadButton>button{border-radius:10px}
[data-testid="stFileUploader"]{border-radius:14px;border:1px dashed #3c657d;background:#10243a}
.hero{border-radius:22px;padding:30px 34px;background:linear-gradient(110deg,#12344b,#123b3a 65%,#124f41);border:1px solid #316460;margin-bottom:22px}
.eyebrow{font-size:.8rem;color:#68e8b2;font-weight:800;text-transform:uppercase;letter-spacing:.17em}
.hero h1{font-size:2.4rem;margin:.3rem 0 .4rem}
.hero p{font-size:1rem;color:#d0e6e6;margin-bottom:0}
.pill{display:inline-block;background:#204837;color:#7dffc1;padding:5px 11px;border-radius:100px;font-weight:700;font-size:.76rem;margin:4px 5px 0 0}
small{color:#9ab0c2}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">PLATAFORMA DE VISIÓN ARTIFICIAL · PERÚ</div><h1>🛣️ Baches Perú</h1><p>Detecta y segmenta daños viales en imágenes y videos con tus modelos entrenados.</p><span class="pill">3 modelos disponibles</span><span class="pill">Segmentación por instancia</span><span class="pill">Exportación de resultados</span></div>''', unsafe_allow_html=True)
with st.sidebar:
    st.markdown('### ⚙️ Centro de control')
    selected = st.selectbox('Modelo de inferencia', list(MODEL_PATHS), help='Cada modelo se carga independientemente desde models/.')
    for n, path in MODEL_PATHS.items():
        st.caption(f"{'🟢' if path.exists() else '🔴'} {n} · {path.name}")
    st.divider()
    conf = st.slider('Confianza mínima', .05, .95, .30, .05)
    iou = st.slider('Umbral IoU (NMS)', .10, .90, .50, .05)
    imgsz = st.select_slider('Resolución de inferencia', options=[320, 416, 512, 640, 768, 960, 1024, 1280], value=640)
    masks = st.toggle('Mostrar máscaras', value=True)
    boxes = st.toggle('Mostrar cajas y etiquetas', value=True)
    alpha = st.slider('Opacidad de máscaras', .10, .85, .42, .05, disabled=not masks)
    st.divider()
    st.caption('Inferencia local en CPU · No se envían archivos a servicios externos. Para acelerar videos, usar un equipo con GPU requiere adaptar device en inference.py.')
params = {"modelo": selected, "confianza": conf, "iou": iou, "imgsz": imgsz, "mascaras": masks, "cajas": boxes, "opacidad": alpha}

t_img, t_vid, t_help = st.tabs(['📸 Imágenes', '🎞️ Video', 'ℹ️ Información'])
with t_img:
    st.subheader('Análisis de imágenes')
    uploaded = st.file_uploader('Adjunta una o varias imágenes', type=['jpg','jpeg','png','bmp','webp'], accept_multiple_files=True, key='imgs')
    if uploaded:
        st.caption(f'{len(uploaded)} imagen(es) preparada(s) · Modelo: {selected}')
        if st.button('🔍 Detectar y segmentar imágenes', type='primary', use_container_width=True):
            outputs = []
            try:
                model = load_model(selected)
                bar = st.progress(0, text='Iniciando inferencia')
                for i, uploaded_img in enumerate(uploaded):
                    original = image_from_bytes(uploaded_img.getvalue())
                    marked, rows, all_masks = process_frame(model, original, conf, iou, imgsz, masks, boxes, alpha)
                    outputs.append((uploaded_img.name, original, marked, rows, all_masks))
                    bar.progress((i+1)/len(uploaded), text=f'Analizadas {i+1}/{len(uploaded)} imágenes')
                st.session_state.image_result = (selected, params.copy(), outputs)
            except Exception as ex:
                st.error(f'No se pudo procesar el archivo: {ex}')
    if 'image_result' in st.session_state:
        used, saved_params, outputs = st.session_state.image_result
        st.success(f'Resultados guardados en esta sesión · Modelo utilizado: {used}')
        total_detected = sum(len(r) for _, _, _, r, _ in outputs)
        a,b,c = st.columns(3)
        a.metric('Imágenes analizadas', len(outputs)); b.metric('Detecciones totales', total_detected); c.metric('Promedio / imagen', f'{total_detected/len(outputs):.1f}')
        for name, original, marked, rows, all_masks in outputs:
            with st.expander(f'📷 {name} · {len(rows)} detecciones', expanded=len(outputs)==1):
                a,b = st.columns(2)
                a.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption='Imagen original', use_container_width=True)
                b.image(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB), caption='Detección y segmentación', use_container_width=True)
                if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else: st.info('Sin detecciones con los parámetros seleccionados. Puedes bajar el umbral de confianza y volver a ejecutar.')
                st.download_button('⬇️ Imagen procesada (PNG)', png_bytes(marked), file_name=f'{Path(name).stem}_segmentado.png', mime='image/png', key=f'im_{name}')
        pack = zip_images([(name, marked, rows, ms) for name, _, marked, rows, ms in outputs], saved_params)
        st.download_button('📦 Descargar resultados completos ZIP (imágenes + máscaras + CSV + JSON)', pack, file_name='baches_peru_imagenes.zip', mime='application/zip', type='primary', use_container_width=True)
with t_vid:
    st.subheader('Análisis de video')
    video = st.file_uploader('Carga un video MP4, AVI, MOV o MKV', type=['mp4','avi','mov','mkv'], key='video')
    frame_step = st.selectbox('Analizar cada N fotogramas', [1,2,3,5,10], index=0, help='1 analiza todos. Los fotogramas omitidos quedan sin anotación; no hay seguimiento de baches únicos.')
    if video:
        st.caption(f'Archivo: {video.name} · {video.size/1024/1024:.1f} MB')
        if st.button('🎬 Procesar video', type='primary', use_container_width=True):
            try:
                model = load_model(selected)
                progress = st.progress(0, text='Abriendo video')
                result = process_video(video.getvalue(), video.name, model, conf, iou, imgsz, masks, boxes, alpha, frame_step, progress)
                st.session_state.video_result = (selected, result)
            except Exception as ex:
                st.error(f'No fue posible procesar el video: {ex}')
    if 'video_result' in st.session_state:
        used, result = st.session_state.video_result
        st.success(f'Video procesado · Modelo: {used} · Codificación: {result["encoding"]}')
        a,b,c,d = st.columns(4)
        a.metric('Fotogramas totales', result['frames']); b.metric('Fotogramas analizados', result['analyzed']); c.metric('Detecciones por fotograma', result['detections']); d.metric('Duración', f'{result["seconds"]:.1f} s')
        st.caption('Las detecciones son observaciones por fotograma: un mismo bache puede contabilizarse varias veces. No representan baches únicos.')
        if result['encoding']=='H.264': st.video(result['video'])
        else: st.info('La previsualización puede no estar soportada por el navegador sin FFmpeg. Descarga el MP4 para visualizarlo en VLC.')
        st.download_button('⬇️ Descargar video procesado MP4', result['video'], file_name='baches_peru_video.mp4', mime='video/mp4', type='primary', use_container_width=True)
        x,y = st.columns(2)
        x.download_button('📊 Descargar detecciones CSV', result['csv'], file_name='baches_peru_video.csv', mime='text/csv', use_container_width=True)
        y.download_button('🧾 Descargar detecciones JSON', result['json'], file_name='baches_peru_video.json', mime='application/json', use_container_width=True)
with t_help:
    st.markdown('''### ¿Cómo funciona?\n1. Elige **S43, S44 o V1** en el panel izquierdo.\n2. Ajusta la confianza, IoU, resolución y visualización de máscaras/cajas.\n3. Carga imágenes o video y pulsa el botón de análisis.\n4. Descarga imágenes, máscaras individuales, CSV/JSON o MP4 anotado.\n\n**Importante:** los tres checkpoints contienen arquitecturas de segmentación YOLO. El campo `area_mascara_px` es superficie en píxeles, no una medida física. Los resultados deben validarse antes de tomar decisiones de mantenimiento vial. El video se procesa localmente y para archivos muy grandes necesitarás RAM, CPU y almacenamiento temporal suficientes. Carga únicamente pesos `.pt` procedentes de fuentes confiables.''')
    st.markdown('**Ubicación de modelos:** `models/exp-s43.pt`, `models/exp-s44.pt`, `models/exp-v1.pt`.')
