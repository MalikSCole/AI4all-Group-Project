# Exporting the trained weights

The Streamlit app loads two files that `Seeded Model.ipynb` produces but that are **not in
this repository** — every `torch.save` in the notebook writes to `/kaggle/working/`, which is
scratch space Kaggle wipes when the session ends.

| File | What it is | Why it's needed |
|---|---|---|
| `final_ordinal_multitask_model.pth` | Packaged checkpoint: weights + metadata | The model |
| `regression_target_scaler.pkl` | The `StandardScaler` fit on train targets | Without it the regression head reports standardized units — "calories: -0.42" |

## Steps

1. Open `Seeded Model.ipynb` on Kaggle and attach the Nutrition5k dataset.
2. **Save & Run All (Commit)** — this persists `/kaggle/working/` as a version output.
   Running interactively and closing the tab does **not**.
3. Open the completed version → **Output** tab.
4. Download both files and place them in `models/` in this repo.
5. `streamlit run app/streamlit_app.py` — the error banner is replaced by the uploader.

If the old session is gone and training re-runs: the notebook is seeded (`SEED=42`), but GPU
training is not bit-for-bit deterministic, so expect accuracy near 74.1% rather than exactly
on it. **Report the number you actually got.**

## Sanity check

```bash
python -c "
import torch
ckpt = torch.load('models/final_ordinal_multitask_model.pth', map_location='cpu', weights_only=False)
print('model     :', ckpt['model_name'])
print('classes   :', ckpt['class_names'])
print('test acc  :', ckpt['test_accuracy'])
print('image size:', ckpt['image_size'])
"
```

## Deploying to Streamlit Community Cloud

Cloud installs from the repo, so a local-only `models/` directory won't be there. Options, in
order of preference:

1. **GitHub Release asset** — upload the two files to a release and have `load_model` download
   and cache them on first run (~15 lines).
2. **Commit them** — the checkpoint is ~13 MB; acceptable for a project that ends in Week 12,
   just make it a decision rather than an accident (and say so if asked).
