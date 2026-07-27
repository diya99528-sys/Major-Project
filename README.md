# DermaScan — AI-Based Skin Lesion & Melanoma Detection

Major Project · Healthcare & Digital Health Track

An end-to-end AI application that classifies dermoscopic skin lesion images as **benign** or **malignant**, built as a CNN (transfer learning, ResNet18) served through a Flask REST API with a web front-end.

## Why this project

Early melanoma detection saves lives, but access to dermatologist screening is limited. This project is a triage-aid prototype: not a diagnostic device, but a demonstration of a real, deployable AI pipeline for this problem. See `docs/Technical_Report.docx` for full justification of every decision.

## Project structure

```
melanoma-project/
├── model/
│   ├── model.py              # ResNet18-based classifier architecture
│   ├── train.py              # training script (works on sample or full data)
│   └── model_weights.pth     # trained weights (demo model)
├── backend/
│   └── app.py                # Flask REST API
├── frontend/
│   └── index.html            # upload UI, calls the API
├── notebooks/
│   └── train_on_ISIC_colab.ipynb   # full training pipeline on the real ISIC dataset
├── data/sample/               # small synthetic demo dataset (see note below)
├── docs/
│   └── Technical_Report.docx  # full technical report
└── requirements.txt
```

## Important note on the dataset

This demo ships with a **small synthetic sample dataset** (120 benign + 120 malignant images), generated programmatically, because the development sandbox used to build this project could not reach external dataset hosts (ISIC Archive / Kaggle) to download the real data. It exists purely so you can run the entire pipeline — train, serve, predict — locally, right now.

**For your actual submission**, train on the real ISIC dataset using `notebooks/train_on_ISIC_colab.ipynb` in Google Colab (free GPU), then drop the resulting `model_weights.pth` into `model/` to replace the demo weights. Cite the dataset:

> Codella, N. et al. "Skin Lesion Analysis Toward Melanoma Detection" (ISIC), International Symposium on Biomedical Imaging (ISBI), 2018. https://www.isic-archive.com/

## Running the project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain the demo model
```bash
cd model
python train.py --data_dir ../data/sample --epochs 8 --no-pretrained --out model_weights.pth
```
> Drop `--no-pretrained` if your machine has internet access to download ImageNet weights — this gives better results even on the small sample.

### 3. Start the backend
```bash
cd backend
python app.py
```
Runs on `http://127.0.0.1:5000`.

### 4. Open the frontend
Open `frontend/index.html` directly in your browser (double-click it, or serve it with any static server). Upload an image and click "Analyze lesion."

## API Reference

See `docs/API_Documentation.md`.

## Academic integrity

The model uses the standard, publicly-documented ResNet18 architecture via transfer learning (standard practice, not copied from any specific tutorial or repo). The sample dataset is synthetically generated for this project. All dataset citations for the full ISIC-based training are listed above and in the technical report.

## Deliverables checklist (per project guidelines)

- [x] GitHub-ready repository structure
- [x] Technical Report (`docs/Technical_Report.docx`)
- [ ] Presentation (PPT) — generate separately if required by your course
- [x] Working demonstration (Flask API + frontend)
- [x] API Documentation (`docs/API_Documentation.md`)
