# ================================================================
# Autor: Milton Murillo Vega (estudiante)
# Origen: Código extraído del chat con ChatGPT (asistente)
# Descripción: Script mínimo que carga un modelo TFLite, ejecuta
#              inferencia sobre una imagen y muestra el Top-5.
# Uso: python3 classify.py   (requiere model.tflite, labels.txt, imagen.jpg)
# ================================================================

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite import Interpreter  # type: ignore

MODEL  = "model.tflite"
LABELS = "labels.txt"
IMAGE  = "imagen.jpg"
TOPK   = 5

# Cargar etiquetas
with open(LABELS, "r", encoding="utf-8") as f:
    labels = [ln.strip() for ln in f if ln.strip()]

# Cargar modelo
interp = Interpreter(model_path=MODEL)
interp.allocate_tensors()

# Entrada
in_det = interp.get_input_details()[0]
in_idx = in_det["index"]
H, W = int(in_det["shape"][1]), int(in_det["shape"][2])
in_dtype = in_det["dtype"]

# Imagen
img = Image.open(IMAGE).convert("RGB")
w, h = img.size
side = min(w, h)
img = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2)).resize((W, H), Image.BILINEAR)
arr = np.asarray(img)

if in_dtype == np.float32:
    x = (arr.astype(np.float32) / 255.0)[None, ...]
else:
    x = arr.astype(in_dtype)[None, ...]

# Inferencia
interp.set_tensor(in_idx, x)
interp.invoke()

# Salida
out_det = interp.get_output_details()[0]
y = np.array(interp.get_tensor(out_det["index"])).squeeze()

# De-cuantización si aplica
scale, zp = out_det.get("quantization", (0.0, 0))
if y.dtype in (np.uint8, np.int8) and scale and scale > 0:
    y = scale * (y.astype(np.float32) - float(zp))

# Softmax si no suma 1
y = y.astype(np.float32)
if not (0.99 <= float(np.sum(y)) <= 1.01):
    y = y - np.max(y)
    y = np.exp(y) / np.sum(np.exp(y))

# Top-K
k = max(1, min(TOPK, y.shape[-1]))
idxs = np.argsort(y)[::-1][:k]

for rank, i in enumerate(idxs, 1):
    label = labels[i] if i < len(labels) else f"class_{i}"
    print(f"{rank}. [{i}] {label}: {float(y[i]):.6f}")
