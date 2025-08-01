# Phase 2: CV Engineer Report

## 1. Detailed Methodology

### 👨‍💻 My Role
As the CV Engineer, I was responsible for training and evaluating image classification models to assess the quality and robustness of our team's produce image dataset. This included two separate tasks:
- **Produce Classification (4-class model)**: Classifying the type of produce
- **Variation Classification (2-class model)**: Identifying the photographic variation (`Whole` vs `Sliced-Cored`)

---

### 🛠️ Step-by-Step Process

#### 📁 Dataset Preprocessing
- Dataset contained images of: `Broccoli`, `Mango`, `Nut`, `Pepper`
- Each produce folder included 3 batch folders, each containing subfolders like `Whole`, `Sliced-Cored`, `In-Context`
- Since `In-Context` images were unavailable, the variation task was adjusted to binary (`Whole`, `Sliced-Cored`)
- I used a custom Python script to:
  - Recursively collect image paths
  - Normalize variation folder names (e.g., sliced_cored, incontext, etc.)
  - Output 4 CSV files: `produce_train.csv`, `produce_val.csv`, `variation_train.csv`, `variation_val.csv`

#### ⚙️ Tools & Libraries
- Framework: **TensorFlow/Keras**
- Model Architecture: **MobileNetV2** (pre-trained on ImageNet)
- Data handling: `ImageDataGenerator` using `.flow_from_dataframe()`
- Training Configuration:
  - Loss:
    - `categorical_crossentropy` for produce classifier
    - `binary_crossentropy` for variation classifier
  - Optimizer: `Adam`
  - Input Size: `224x224x3`
  - Batch Size: 64
  - Epochs: 3–5
  - Early Stopping: Enabled

---

### 🤖 Model Training

#### 1. Produce Classifier
- **Goal**: Classify `Broccoli`, `Mango`, `Nut`, `Pepper` (4 classes)
- Used MobileNetV2 as a frozen base model with a custom classification head
- Trained using softmax output and categorical crossentropy
- Early stopping triggered after 2–3 epochs

#### 2. Variation Classifier
- **Goal**: Classify variation type as `Whole` or `Sliced-Cored`
- Due to limited data, this was treated as a **binary classification** task
- Used a sigmoid output neuron and binary crossentropy loss
- Trained quickly (under 15 minutes) with early stopping enabled

---

## 2. Analysis & Justifications

### 🔍 Model Choices
- **MobileNetV2**: Chosen for fast training and high accuracy, especially on mid-sized image datasets
- **Binary variation classifier**: Necessary due to absence of `In-Context` class
- **CSV-based flow_from_dataframe()** allowed precise control over label splits and file mapping

---

### 📊 Model Performance

#### Produce Classifier (4-class)
- **Final Validation Accuracy**: ~96%
- **Confusion Matrix Highlights**:
  - `Broccoli` and `Nut` had very high precision
  - Some confusion between `Mango` and `Pepper` likely due to similar colors/textures in certain lighting

#### Variation Classifier (2-class)
- **Final Validation Accuracy**: ~80%
- **Confusion Matrix Highlights**:
  - Model strongly predicted `Sliced-Cored` when clearly visible
  - A few `Whole` images under unusual angles were misclassified

---

### 📈 Visualizations
- **Confusion Matrices** for both models were generated using `sklearn.metrics` and `seaborn`
- (Optional) Accuracy/loss curves could be included by saving Keras `fit()` history
- Files saved:
  - `confusion_matrix_produce_classifier_mobilenetv2.png`
  - `confusion_matrix_variation_classifier_mobilenetv2.png`

---

## 3. Insights & Reflections

### ✅ What Worked Well
- MobileNetV2 proved effective and fast to train
- Binary variation classifier performed reliably even with class imbalance
- Script-based preprocessing ensured consistency and reproducibility

### ⚠️ Challenges
- Missing `In-Context` class reduced complexity of variation classification
- Some folders had non-standard names, requiring robust label extraction
- Class balance needed manual validation after splitting

### 🔁 Future Improvements
- Add more `In-Context` images to re-enable 3-class variation classifier
- Fine-tune MobileNetV2 layers for better generalization
- Use `ImageDataGenerator` for on-the-fly data augmentation

---

## 📦 Output Artifacts
- `produce_classifier_mobilenetv2.h5`
- `variation_classifier_binary_mobilenetv2.h5`
- `produce_train.csv`, `variation_train.csv`, and corresponding validation files
- `produce_classes.csv`, `variation_classes.csv`
- `confusion_matrix_produce_classifier_mobilenetv2.png`
- `confusion_matrix_variation_classifier_mobilenetv2.png`

---

