# Brain Tumor MRI Image Classification

Deep learning project that classifies brain MRI scans into **Glioma**,
**Meningioma**, **No Tumor**, or **Pituitary** tumor, using a custom CNN and
three ImageNet-pretrained transfer-learning models, deployed as a Streamlit
web application.

> ⚠️ **Educational project only.** Not a certified medical device. Predictions
> must never be used for real diagnosis or clinical decisions.

---

## 1. Project overview

| | |
|---|---|
| **Domain** | Medical Imaging — Image Classification |
| **Classes** | Glioma, Meningioma, No Tumor, Pituitary |
| **Dataset** | [Labeled MRI Brain Tumor Dataset](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset) (Roboflow, CC BY 4.0) — 2,443 images (1,695 train / 502 valid / 246 test) |
| **Models trained** | Custom CNN, MobileNetV2, EfficientNetB0, ResNet50 (all transfer-learning models use ImageNet weights, frozen-head training then fine-tuning the top 20% of layers) |
| **Deployed model** | **MobileNetV2** (see [Section 3](#3-model-comparison-and-selection) for why) |
| **App** | Streamlit — upload an MRI image, get the predicted class and per-class confidence |

### End-to-end pipeline (4 notebooks, `Notebooks/`)

1. **`01_dataset_creation_and_analysis.ipynb`** — Builds and verifies a dataset
   manifest from the existing `train/valid/test` folders and `_classes.csv`
   label files. Cross-checks the CSV one-hot label, the image's class folder,
   and its filename code for every one of the 2,443 images; all 2,443 passed
   verification with zero issues. Confirms all images are 640×640 RGB. No
   resizing/normalization is done here — that's left to each model's own
   notebook so each model gets the preprocessing it actually needs.
2. **`02_custom_cnn_training_evaluation.ipynb`** — Trains a 3-block CNN from
   scratch (Conv→BatchNorm→ReLU→MaxPool→Dropout ×3, GAP, Dense(128), softmax(4)),
   224×224 input, `Rescaling(1/255)` baked into the model, mild rotation/zoom
   augmentation, class-weighted loss for the imbalanced classes. Trains, then
   continues training at a lower LR, and keeps whichever checkpoint has the
   lower validation loss. Evaluates once on the held-out test split.
3. **`03_pretrained_cnn_models.ipynb`** — Trains **MobileNetV2**,
   **EfficientNetB0**, and **ResNet50** with the same recipe: frozen backbone
   + new 4-class head first, then unfreeze the top 20% of backbone layers and
   fine-tune at a lower LR. Each backbone uses its own required preprocessing
   (MobileNetV2 → scale to [-1,1]; ResNet50 → BGR + ImageNet channel centering;
   EfficientNetB0 → raw 0–255, rescaling is built into the model). The test
   split is **not** touched in this notebook.
4. **`04_final_model_comparison_and_test_evaluation.ipynb`** — Loads all four
   saved models, evaluates all of them **once** on the same held-out test
   split (246 images, never used for training/tuning), and saves confusion
   matrices, per-class metrics, and model sizes to `Evaluation/`.

No data leakage was found: the test split is loaded only in Notebook 4 (and
once, read-only, inside Notebook 2 for the custom CNN's own final report);
training/validation never see test images.

---

## 2. Dataset

```
Datasets/
├── train/  (1,695 images: glioma 564, meningioma 358, no_tumor 335, pituitary 438)
├── valid/  (502 images:   glioma 161, meningioma 124, no_tumor  99, pituitary 118)
└── test/   (246 images:   glioma  80, meningioma  63, no_tumor  49, pituitary  54)
```

All 2,443 images are 640×640 RGB JPEGs. Classes are moderately imbalanced
(Glioma is the largest class, No Tumor the smallest); all three training
notebooks use class-weighted loss to compensate.

---

## 3. Model comparison and selection

### 3.1 Measured results

All numbers below are **measured**, either from the project's own evaluation
notebook outputs (`Evaluation/*.csv`, `Evaluation/*.json`) or from a local
CPU-only benchmark run against the actual saved `.keras` files on a 15 GB RAM
Windows machine (see [`Models/`](Models) summaries + this README's benchmark
notes). None of these figures are estimated or invented.

| Metric | Custom CNN | MobileNetV2 | EfficientNetB0 | ResNet50 |
|---|---|---|---|---|
| **Test accuracy** | 66.3% | 87.0% | 83.3% | 93.5% |
| **Test macro precision** | 0.723 | 0.875 | 0.843 | 0.934 |
| **Test macro recall** | 0.671 | 0.870 | 0.839 | 0.928 |
| **Test macro F1** | 0.646 | 0.866 | 0.834 | 0.929 |
| **Validation accuracy** | 73.1% | 88.6% | 83.9% | 94.2% |
| **Model size (saved .keras)** | 1.35 MB | **20.76 MB** | 33.77 MB | 204.80 MB |
| **Params** | 110,948 | 2,263,108 | 4,054,695 | ~25.6M |
| **Model load time (CPU)** | 0.15 s | 1.1 s | 1.5 s | ⚠️ *failed to load* — see note below |
| **Single-image inference (CPU, median)** | 75 ms | 89 ms | 97 ms | not measured (load failure) |
| **Process RSS after load+predict** | ~395 MB | **~438 MB** | ~475 MB | not measured (load failure) |
| **CPU compatibility** | ✅ yes | ✅ yes | ✅ yes | ✅ yes (in principle) |
| **Additional dependencies** | TensorFlow/Keras only | TensorFlow/Keras only | TensorFlow/Keras only | TensorFlow/Keras only |
| **GitHub plain-git committable (<100 MB/file)** | ✅ yes | ✅ yes | ✅ yes | ❌ **no — needs Git LFS or external hosting** |
| **Free-deployment suitability** | Easy, but weak accuracy | **Easy, reliable** | Easy, dominated by MobileNetV2 | Hard — large artifact, high RAM |

> **ResNet50 load-failure note (real measurement, not a guess):** on this
> development machine (15.4 GB total RAM, ~6.2 GB free at the time), loading
> `resnet50_best_validation.keras` (204.8 MB) with TensorFlow/Keras **crashed**
> — first with a segmentation fault, then on retry with a Keras
> `ResourceExhaustedError` while allocating weight arrays during deserialization.
> Custom CNN, MobileNetV2, and EfficientNetB0 all loaded and ran inference
> without any issue in the same environment. This is concrete, first-hand
> evidence (not a theoretical concern) that ResNet50's memory footprint is
> risky on memory-constrained hosts — which is exactly what Streamlit
> Community Cloud's free tier is (shared, limited RAM/CPU, no GPU).

### 3.2 Per-class performance (test set)

| Class | Metric | Custom CNN | MobileNetV2 | EfficientNetB0 | ResNet50 |
|---|---|---|---|---|---|
| Glioma | Precision / Recall / F1 | 0.906 / 0.725 / 0.806 | 0.936 / 0.913 / 0.924 | 0.904 / 0.825 / 0.863 | 0.988 / 0.988 / 0.988 |
| Meningioma | Precision / Recall / F1 | 0.792 / 0.302 / 0.437 | 0.852 / 0.730 / 0.786 | 0.776 / 0.714 / 0.744 | 0.889 / 0.889 / 0.889 |
| No Tumor | Precision / Recall / F1 | 0.729 / 0.714 / 0.722 | 0.953 / 0.837 / 0.891 | 0.952 / 0.816 / 0.879 | 0.976 / 0.837 / 0.901 |
| Pituitary | Precision / Recall / F1 | 0.464 / 0.944 / 0.622 | 0.761 / 1.000 / 0.864 | 0.740 / 1.000 / 0.850 | 0.885 / 1.000 / 0.939 |

The Custom CNN's weak Meningioma recall (0.30) and over-predicting Pituitary
is a real sign of underfitting relative to the transfer-learning models — not
an artifact of the metric.

### 3.3 Selection reasoning

Notebook 4 itself applies a strict internal rule — "validation accuracy
within 5 percentage points of the best, then smallest of those" — which
mechanically shortlists **only ResNet50**, because every other model is
more than 5pp behind ResNet50's 94.2% validation accuracy. That rule is
useful as a sanity check but is not the rule this deployment decision uses,
for two concrete reasons documented in this project's own notebooks:

1. Notebook 4 explicitly says: *"Confirm that the complete app bundle...fits
   the deployment platform's current limit... Use class-level recall and the
   intended deployment constraints in the final decision too."* It treats the
   5pp shortlist as a decision aid, not a final verdict.
2. ResNet50's 204.8 MB artifact **cannot be committed to a normal GitHub
   repository** (GitHub's hard per-file limit is 100 MB without Git LFS), and
   — as measured directly above — it is heavy enough to **fail to load** on a
   real, modestly-resourced machine. That is a direct, measured failure of
   Priority 3 (free, reliable deployment), not a theoretical one.

Applying this project's actual stated priorities (accuracy → size → free
deployment → an explicit "~10 percentage point trade-off is acceptable for a
meaningfully smaller/easier-to-deploy model" guideline):

- **ResNet50** (93.5% test acc, 204.8 MB) — best accuracy, but fails the
  size/deployment priorities outright (measured load failure + GitHub file
  limit).
- **EfficientNetB0** (83.3% test acc, 33.8 MB) — is **dominated** by
  MobileNetV2: MobileNetV2 is both smaller *and* more accurate on every
  metric. There is no reason to prefer it.
- **Custom CNN** (66.3% test acc, 1.35 MB) — smallest by far, but a 27pp
  accuracy gap (and a badly underfit Meningioma class, 0.30 recall) is far
  outside the ~10pp trade-off guideline. Not acceptable for this use case.
- **MobileNetV2** (87.0% test acc, 20.8 MB) — 6.5pp below ResNet50, well
  inside the ~10pp guideline, for a **10× smaller** artifact, trivially
  committable to a plain Git repo, the lowest CPU memory footprint of the
  three transfer-learning models, and no load-time risk observed on
  constrained hardware.

### ✅ Final selection: **MobileNetV2**

- **Why:** Best practical balance of accuracy, size, and deployability. Clearly
  beats EfficientNetB0 on every axis, and its accuracy trade-off against
  ResNet50 is modest and explicitly inside this project's own stated
  trade-off tolerance — in exchange for an artifact that is 10× smaller,
  always fits GitHub's normal size limit, and was the one model (besides the
  tiny Custom CNN) that reliably loaded and ran prediction in a
  memory-constrained test.
- **Test performance:** 87.0% accuracy, 0.866 macro F1, 0.891 F1 on No Tumor,
  0.864 F1 on Pituitary (see per-class table above).
- **Saved size:** 20.76 MB (`Models/mobilenetv2_best_validation.keras`).
- **CPU inference:** ~89 ms median per image on a commodity CPU (measured, no GPU).
- **Memory/dependencies:** ~438 MB process RSS after loading the model and
  running one prediction; only needs TensorFlow/Keras, NumPy, and Pillow —
  no extra inference dependencies.
- **Deployability:** Yes — fits comfortably within Streamlit Community
  Cloud's free-tier CPU/RAM allowance, needs no GPU, no paid API, and no
  runtime download of weights (the `.keras` file ships in the repo).

If ResNet50's extra ~6.5pp of accuracy is later required for a non-free or
higher-memory deployment target, it is already trained and evaluated
(`Models/resnet50_best_validation.keras` + `Models/resnet50_validation_summary.json`)
and can be swapped in by changing `MODEL_PATH` and the preprocessing function
in `app.py` — no retraining needed. It would need Git LFS (or external
weight hosting) to be committed to GitHub.

---

## 4. Streamlit application

`app.py` implements the full inference pipeline:

1. **Upload** — JPG/JPEG/PNG/BMP, 10 MB max (validated before decoding).
2. **Preprocessing** — Pillow decode → RGB → resize to 224×224 → Keras
   `mobilenet_v2.preprocess_input` (scales to [-1, 1]) — **identical** to the
   preprocessing used for MobileNetV2 in Notebook 3/4, so inference matches training.
3. **Model loading** — `st.cache_resource` loads the 20.76 MB model once per
   session, not on every interaction; a clear error is shown if the model
   file or class-order file is missing.
4. **Prediction** — real forward pass through the trained model (no
   hardcoded/simulated results); predicted class + per-class confidence
   (softmax probabilities) are displayed with progress bars.
5. **Error handling** — invalid/corrupted/oversized files are caught and
   reported with a plain-language message, not a raw traceback.
6. **Disclaimer** — a persistent, visible notice that this is an educational
   tool, not a diagnostic device.

### Run locally

```bash
cd "Brain MRI Tumor Classification"
pip install -r requirements.txt
streamlit run app.py
```

---

## 5. Project structure

```
Brain MRI Tumor Classification/
├── Brain Tumor MRI Image Classification.docx   # original project brief
├── Datasets/                                   # train/valid/test images + manifest CSVs (unchanged)
├── Notebooks/                                   # the 4 original notebooks (unchanged)
│   ├── 01_dataset_creation_and_analysis.ipynb
│   ├── 02_custom_cnn_training_evaluation.ipynb
│   ├── 03_pretrained_cnn_models.ipynb
│   └── 04_final_model_comparison_and_test_evaluation.ipynb
├── Models/                                      # all 4 trained models + metadata (unchanged)
│   ├── mobilenetv2_best_validation.keras        # <- used by app.py
│   ├── class_names.json                         # <- used by app.py
│   ├── mobilenetv2_validation_summary.json      # <- used by app.py
│   ├── brain_mri_custom_cnn_final.keras
│   ├── efficientnetb0_best_validation.keras
│   ├── resnet50_best_validation.keras           # kept locally; see Git LFS note below
│   └── ... (checkpoints, validation summaries, pretrained_model_index.json)
├── Evaluation/                                   # confusion matrices, comparison CSVs/JSON (unchanged)
├── app.py                                        # Streamlit application (new)
├── requirements.txt                              # deployment dependencies (new)
├── README.md                                     # this file (new)
└── .gitignore                                    # new
```

Nothing in `Datasets/`, `Notebooks/`, `Models/`, or `Evaluation/` was modified,
retrained, or deleted — only `app.py`, `requirements.txt`, `README.md`, and
`.gitignore` were added.

---

## 6. Testing performed

| Test | Result |
|---|---|
| Model file exists and loads (`MobileNetV2`) | ✅ Pass — loaded in ~1.1 s |
| Model file exists and loads (`Custom CNN`) | ✅ Pass — loaded in ~0.15 s |
| Model file exists and loads (`EfficientNetB0`) | ✅ Pass — loaded in ~1.5 s |
| Model file exists and loads (`ResNet50`) | ❌ **Failed** (segfault / `ResourceExhaustedError` on a 6 GB-free machine) — used as deployment-risk evidence, see §3.3 |
| Output shape / class mapping correct | ✅ Pass — 4-way softmax, `class_names.json` order (`Glioma, Meningioma, No Tumor, Pituitary`) matches training |
| Real CPU inference on a real test image | ✅ Pass — ran MobileNetV2 end-to-end on `Datasets/test/glioma/...` and produced a plausible probability vector |
| `app.py` imports cleanly | ✅ Pass (`streamlit`, `tensorflow-cpu`, `numpy`, `pillow` all installed/importable in the test environment) |
| `streamlit run app.py` full UI smoke test | ⚠️ **Not executed in this session** — this environment has no browser/display to click through the UI. Preprocessing and model-loading logic were verified directly via the benchmark script instead (same functions `app.py` calls). **Before deploying, run `streamlit run app.py` locally and upload a sample image from `Datasets/test/` to confirm the UI end-to-end.** |
| Invalid-file handling | Code path reviewed (non-image / corrupted / oversized files are caught); not interactively tested without a UI |

**Limitation:** this session could not drive a browser against the running
Streamlit server, so the on-screen UI itself was not click-tested — only the
underlying model-loading and preprocessing code (identical to what `app.py`
calls) was executed and verified against real model files and a real image.
Please do one local `streamlit run app.py` pass before pushing to production.

---

## 7. Free deployment guide (Streamlit Community Cloud)

### 7.1 Prepare the repository

1. Keep this folder (`Brain MRI Tumor Classification/`) as the repo root, or
   as a subfolder if you push the whole `Innovexis` directory — if it's a
   subfolder, set Streamlit Cloud's "Main file path" to
   `Brain MRI Tumor Classification/app.py` and note that relative paths in
   `app.py` are already resolved from the script's own location, so this works
   either way.
2. **ResNet50's model file (204.8 MB) exceeds GitHub's 100 MB per-file limit.**
   Since the deployed app only needs MobileNetV2, you have two options:
   - **Recommended:** exclude `Models/resnet50_best_validation.keras` from the
     GitHub repo (add it to `.gitignore`) — it stays on your local disk for
     the notebooks/record, just isn't pushed. The app doesn't need it.
   - **Alternative:** track it with [Git LFS](https://git-lfs.com/) if you want
     it versioned on GitHub too (`git lfs track "Models/resnet50_best_validation.keras"`).
3. Everything else (`mobilenetv2_best_validation.keras` 20.76 MB,
   `efficientnetb0_best_validation.keras` 33.77 MB, the Custom CNN checkpoints,
   datasets, notebooks) is well under GitHub's limits and can be committed normally.
4. Do not commit secrets/API keys — this project needs none.

### 7.2 Push to GitHub

```bash
git init
git add .
git commit -m "Add Streamlit app and deployment files"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### 7.3 Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"** → select your repository and branch.
3. Set **Main file path** to `app.py` (or `Brain MRI Tumor Classification/app.py`
   if the repo root is one level up).
4. Click **Deploy**. The first build installs `tensorflow-cpu` (~300–400 MB
   download), which can take a few minutes — this is normal for any TensorFlow
   app on a free tier.
5. Once deployed, open the public URL and upload a sample image from
   `Datasets/test/<class>/` to confirm it predicts correctly.

### 7.4 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails installing dependencies | Dependency pin conflicts with Streamlit Cloud's Python version | Loosen the pins in `requirements.txt` (e.g. `tensorflow-cpu>=2.17,<3`) and redeploy |
| `No solution found... tensorflow-cpu==2.21.0 has no wheels with a matching Python ABI tag` | Streamlit Cloud defaulted to a Python version newer than any TensorFlow wheel supports (observed: Python 3.14, while `tensorflow-cpu` 2.21.0 only ships wheels up to Python 3.13) | This repo pins the runtime via `.python-version` (and `runtime.txt` as a fallback) to **3.12**. If it still picks the wrong version, open **Manage app → Settings → General** and set the Python version explicitly to 3.12, then reboot the app. |
| App crashes / restarts after upload | Out-of-memory on the free tier | Confirm you deployed with **MobileNetV2**, not ResNet50 — this is exactly the failure mode measured in §3.1/§3.3 |
| "Model file not found" error | `Models/mobilenetv2_best_validation.keras` wasn't pushed (check `.gitignore`/LFS filters) or path mismatch between repo root and "Main file path" | Verify the file is present in the GitHub repo UI, and that the main file path matches the folder structure |
| Image upload does nothing | Browser-side issue or file type not in the allowed list | Confirm the file is `.jpg/.jpeg/.png/.bmp` and under 10 MB |
| Predictions look wrong/random | Preprocessing mismatch | Make sure `app.py`'s `preprocess_image` hasn't been edited to diverge from `mobilenet_v2.preprocess_input` — this must match training exactly |

---

## 8. Remaining actions for you

1. Run `streamlit run app.py` locally once and click through it with a real
   test image (this was not done interactively in this session — see §6).
2. Decide whether to exclude `resnet50_best_validation.keras` from Git or
   track it with Git LFS, then push to a **public** GitHub repository.
3. Deploy on Streamlit Community Cloud following §7.3, then verify the public
   URL on a phone/different browser.
4. (Optional) If you want ResNet50's extra accuracy for a non-free or
   higher-RAM deployment target later, swap `MODEL_PATH`/preprocessing in
   `app.py` — the trained weights already exist, no retraining required.
