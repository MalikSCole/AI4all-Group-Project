# Food Image Calorie Estimator

This project uses the **Nutrition5k** dataset to explore whether computer vision can estimate the calorie range of a meal from a food image. Instead of predicting exact calories, the model classifies food images into three categories:

- **Low calorie**
- **Medium calorie**
- **High calorie**

The project was developed as part of the **AI4ALL AI Fellowship** and includes data exploration, CNN modeling, model evaluation, data leakage checking, and Streamlit deployment.

---

## Dataset

**Dataset:** Nutrition5k Dataset  
**Source:** Kaggle / Google Research  
**Link:** https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset/data

The dataset contains food images and dish-level nutrition labels such as calories, mass, fat, carbohydrates, and protein.

The dataset is not included in this repository because of file size. To run the notebooks locally, download the dataset and place it in:

```text
data/nutrition5k/
```

For Kaggle notebooks, the dataset can be added directly through the Kaggle **Add Data** panel.

---

## Modeling Approach

The model uses overhead RGB food images from Nutrition5k. Calorie values are grouped into three balanced classes:

```text
Low / Medium / High
```

The dataset is split into:

- **70% training**
- **15% validation**
- **15% testing**

The validation set is used for model comparison and refinement, while the test set is reserved for final evaluation.

---

## Model Evaluation

The team evaluates model performance using classification metrics and visualizations, including:

- accuracy
- precision
- recall
- specificity
- F1-score
- confusion matrix
- training loss and accuracy curves
- ROC/AUC visualization

Evaluation artifacts are stored in the `evaluation/` folder.

A dish-level data leakage check was also completed. The check confirmed **0 overlapping dishes** across the train, validation, and test sets, supporting a cleaner evaluation process for the current overhead RGB image setup.

---

## Streamlit Deployment

A Streamlit app is included in the `app/` folder. The app allows users to upload a food image and receive a predicted calorie range.

App flow:

```text
Upload food image → Preprocess image → Run model → Display Low / Medium / High prediction
```

The app also includes:

- image preview
- ordinal threshold probabilities
- Grad-CAM visualization
- auxiliary nutrition estimates
- model caveats and limitations

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app expects trained model files in the `models/` folder:

```text
models/final_ordinal_multitask_model.pth
models/regression_target_scaler.pkl
```

If these files are missing, see `EXPORT_WEIGHTS.md` for how to export them from the training notebook.

---

## Repository Structure

```text
AI4all-Group-Project/
├── app/                              # Streamlit deployment app
├── evaluation/                       # Model evaluation scripts and plots
├── models/                           # Trained model weights and scaler
├── notebooks/                        # Data exploration and modeling notebooks
├── plots/                            # Dataset exploration visualizations
├── src/                              # Supporting Python scripts and model code
├── dish-level-data-leakage-check.ipynb
├── FoodImageCalorieEstimator.ipynb
├── Seeded Model.ipynb
├── EXPORT_WEIGHTS.md
├── requirements.txt
└── README.md
```

---

## Limitations

This project is a prototype and should not be used as a medical or dietary tool. The model is trained on Nutrition5k, which was collected in a controlled setting, so it may not generalize well to all real-world food photos.

The model may be less reliable for:

- side-angle phone photos
- poor lighting
- multiple dishes in one image
- unfamiliar cuisines
- cluttered backgrounds
- hidden ingredients, oils, or sauces

The model works best on clear overhead images of single food plates, which are most similar to the training data.

---

## Future Improvements

Possible next steps include:

- testing more real-world food photos
- adding rotation and brightness augmentation
- tuning hyperparameters using the validation set
- improving ROC/AUC evaluation with actual model probabilities
- exploring RGB + depth multimodal modeling
- improving the Streamlit interface and deployment workflow

---

## Team

This project was completed as part of the **AI4ALL AI Fellowship**.

Team members:

- Malik Cole
- Kai Nguyen
- Chidera Onyebu
- Shyam Dudhat
