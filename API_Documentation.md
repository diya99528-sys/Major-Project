# DermaScan API Documentation

Base URL (local): `http://127.0.0.1:5000`

---

## GET /api/health

Health check. Confirms the model loaded correctly and reports which device (CPU/GPU) it's running on.

**Response `200`**
```json
{
  "status": "ok",
  "classes": ["benign", "malignant"],
  "device": "cpu"
}
```

---

## POST /api/predict

Classifies a skin lesion image as benign or malignant.

### Request

Two supported formats:

**Option A — multipart/form-data (recommended, used by the included frontend)**
```
POST /api/predict
Content-Type: multipart/form-data

image: <file>
```

**Option B — JSON with base64-encoded image**
```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Response `200`
```json
{
  "prediction": "malignant",
  "confidence": 97.42,
  "probabilities": {
    "benign": 2.58,
    "malignant": 97.42
  },
  "disclaimer": "This is an educational demo model trained on synthetic sample data, NOT a validated medical diagnostic tool."
}
```

### Response `400` — no image provided
```json
{ "error": "No image provided. Send multipart 'image' or JSON 'image_base64'." }
```

### Response `500` — inference error
```json
{ "error": "<exception message>" }
```

### Example (curl)
```bash
curl -X POST -F "image=@lesion.jpg" http://127.0.0.1:5000/api/predict
```

### Example (JavaScript / fetch)
```javascript
const formData = new FormData();
formData.append("image", fileInputElement.files[0]);

const res = await fetch("http://127.0.0.1:5000/api/predict", {
  method: "POST",
  body: formData,
});
const result = await res.json();
```
