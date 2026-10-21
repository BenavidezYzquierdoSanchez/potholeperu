# Baches Perú · Detector y segmentador web

Aplicación Streamlit para detección y segmentación de daños viales con selector de los tres modelos proporcionados: `exp-s43.pt`, `exp-s44.pt`, `exp-v1.pt`.

## Requisitos
- Python 3.11 o 3.12 (64 bits) recomendado.
- Windows, macOS o Linux. CPU suficiente; procesar videos largos requiere recursos adicionales.
- FFmpeg opcional (recomendado) para exportar MP4 H.264 compatible con navegador.

## Instalación en VS Code

1. Extrae el ZIP y abre la carpeta `baches_peru_web` con VS Code.
2. Abre una terminal en esa carpeta e instala dependencias:

**Windows PowerShell**:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**macOS/Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Abre la dirección que muestra Streamlit (normalmente `http://localhost:8501`). La primera carga del modelo puede tardar algo más.

## Uso
- Selecciona un modelo, ajusta confianza/IoU, resolución y opciones de visualización.
- En **Imágenes**, selecciona JPG/PNG/BMP/WebP (una o varias), procesa, compara, descarga PNG o ZIP con máscaras y CSV/JSON.
- En **Video**, carga MP4/AVI/MOV/MKV, selecciona cada cuántos fotogramas analizar, procesa y exporta MP4/CSV/JSON.
- Para video, el contador de detecciones no representa baches únicos: el mismo bache reaparece en fotogramas sucesivos.
- Área de máscara medida en píxeles; no convierte a metros cuadrados sin calibración.
- Si `FFmpeg` no está instalado, el MP4 exportado utiliza MP4V, que algunos navegadores no reproducen. Instalar FFmpeg y agregarlo al PATH habilita H.264.
- Solo cargar pesos confiables: los `.pt` se deserializan al cargarse.

## Despliegue
Para publicación puedes usar un servidor con Python (VM/cloud), incluyendo AWS EC2. No subas videos/imágenes personales a repositorios públicos. Estos checkpoints ocupan aprox. 69 MB en total, por lo que conviene comprobar los límites de hosting y Git. Instala los requisitos, abre el puerto de Streamlit mediante reverse proxy HTTPS, añade autenticación si habrá datos sensibles y usa almacenamiento seguro. La inferencia predeterminada es por CPU.

## Estructura
```
app.py                 interfaz
inference.py           modelos, inferencia y exportación
models/                tres checkpoints .pt
.streamlit/config.toml apariencia y tamaño de carga
requirements.txt       dependencias
```
