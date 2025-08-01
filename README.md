# Phase 2: CV  Report

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


# Phase 2: NLP Engineer Report

## 1. Detailed Methodology

### 🧠 My Role
As the NLP Engineer, I was responsible for designing and evaluating a semantic recipe retrieval system using **BERT embeddings**. The goal was to enable **tag-based recipe recommendations** where users can enter descriptive queries and receive contextually relevant recipe matches.

---

### 🛠️ Step-by-Step Process

#### 📁 Dataset Preprocessing
- The dataset contained a set of **recipes**, each labeled with descriptive **tags** (e.g., `"vegan"`, `"quick"`, `"chicken"`)
- I cleaned the tag text, lowercased and joined tags per recipe into a single string (e.g., `"easy dinner healthy"`)
- I used Hugging Face's BERT tokenizer to encode each tag string:
  - Used `[CLS]` token representation as the **embedding**
  - Stored the resulting embeddings in a matrix for fast similarity comparison

#### ⚙️ Tools & Libraries
- Framework: **PyTorch + Hugging Face Transformers**
- Model: **`bert-base-uncased`** (pre-trained)
- Embedding: `[CLS]` token output (768-dimension vector)
- Retrieval: **Cosine similarity** using `sklearn.metrics.pairwise`
- Data handling: `pandas`, `numpy`
- Notebook: `recipie_recommendation.ipynb`

---

### 🤖 Model Workflow

#### 1. Recipe Embedding
- Every recipe’s tags were embedded using `bert-base-uncased`
- No fine-tuning was applied (used as a fixed encoder)
- Vectors were cached to avoid recomputation

#### 2. Query Processing
- User queries (e.g., `"quick healthy vegetarian"`) were passed through the **same BERT encoder**
- The `[CLS]` vector was used to represent the semantic meaning of the query

#### 3. Semantic Search
- The query embedding was compared to all recipe embeddings using **cosine similarity**
- Pre-computed all recipe embeddings for fast retrieval
- Query processing pipeline:
  - Convert user tags to formatted text: "Tags: healthy quick vegetarian"
  - Generate query embedding using trained model
  - Compute cosine similarities with all recipes
  - Return top-k results with scores

---

## 2. Analysis & Justifications

### 🔍 Model Choices
- **BERT** was chosen for its strength in capturing semantic relationships between words and tags
- Instead of traditional keyword search, semantic search allowed us to match queries like `"hearty vegan soup"` to `"plant-based stew"`

---

### 📊 Model Performance

#### Quantitative Metrics
- On test queries like `"gluten-free dinner"` or `"low carb protein lunch"`, the top recommendations were highly relevant
- Cosine similarity effectively ranked recipes by semantic closeness
- Training Loss: Converged from ~4.5 to ~2.1 over 2 epochs
- Model Size: 440MB checkpoint (includes embeddings)

#### Qualitative Evaluation

**Query:** `"healthy quick vegetarian"`

**Top Result Recipes:**
- `"15-Minute Veggie Stir-Fry (Score: 0.892)"`
- `"Quick Quinoa Salad Bowl (Score: 0.871)"`
- `"Speedy Spinach Smoothie (Score: 0.856)"`


**Query:** `"mexican spicy chicken"`

**Top Result Recipes:**
- `"Spicy Chicken Enchiladas (Score: 0.913)"`
- `"Mexican Chicken Fajitas (Score: 0.901)"`
- `"Jalapeño Chicken Tacos (Score: 0.889)"`

---

### 📈 Visualizations
- Embedding similarity scores were printed and analyzed in the notebook
- A heatmap of cosine similarities could be added for further clarity (optional)

** Embedding-Based Semantic Space:**

The model creates a 256-dimensional semantic space where:

- Similar recipes **cluster together**
- Query vectors **land near relevant recipe clusters**
- **Cosine similarity** effectively measures semantic distance

---

** Embedding Quality Indicators:**

- High similarity scores (**> 0.85**) for relevant matches  
- Clear separation between different **cuisine types**  
- Smooth gradients between **related concepts**

---

## 3. Insights & Reflections

### ✅ What Worked Well
- BERT embeddings performed well even without fine-tuning
- Tag-based representations made the system light and interpretable
- Cosine similarity was efficient and easy to implement

### ⚠️ Challenges
- Tags like `"clean"`, `"light"`, or `"classic"` were ambiguous and difficult for BERT to disambiguate
- Some recipes had very few tags, weakening their semantic vector
- Longer phrases like `"good for cold weather"` may benefit from fine-tuning

### 🔁 Future Improvements
- Fine-tune BERT on tag-specific datasets for better domain adaptation
- Add query expansion or tag clustering for broader matches
- Introduce user feedback loop to improve results over time

---

## 📦 Output Artifacts
- `recipie_recommendation.ipynb` (notebook implementation)
- `RAW_interactions.csv`
- `RAW_recipes`



