import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import ast
import json

# ============= IMAGE CLASSIFICATION SECTION =============

class ProduceClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, freeze_backbone=False):
        super(ProduceClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
            print("Backbone frozen - only training final classification layer")
    
    def forward(self, x):
        return self.resnet(x)

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to load a model
def load_model(model_path, num_classes, class_names):
    try:
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found.")
            return None, class_names
            
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = ProduceClassifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ Model {model_path} loaded successfully!")
        return model, class_names
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None, class_names

# Load produce classifier
produce_model_path = 'model/best_produce_classifier.pth'
produce_class_names = ['Broccoli', 'Mango', 'Nut', 'Pepper']
produce_model, produce_class_names = load_model(produce_model_path, num_classes=4, class_names=produce_class_names)

print(f"Produce class names: {produce_class_names}")

# Load variation classifiers
variation_model_map = {
    'Mango': 'model/mango_variation_classifier.pth',
    'Pepper': 'model/pepper_variation_classifier.pth',
    'Nut': 'model/nut_variation_classifier.pth',
    'Broccoli': 'model/broccoli_variation_classifier.pth'
}

variation_class_names = {
    'Broccoli': ['Florets', 'In-Context(Cooking)', 'Whole Crown'],
    'Mango': ['Cubed,Hedgehog', 'Sliced,Peeled', 'Whole'],
    'Nut': ['chopped', 'sliced', 'whole'],
    'Pepper': ['Diced,Sliced', 'Halved,Deseeded', 'Whole']
}

variation_models = {}

for produce, path in variation_model_map.items():
    if os.path.exists(path):
        model, _ = load_model(path, num_classes=3, class_names=variation_class_names[produce])
        variation_models[produce] = model
    else:
        print(f"Variation model {path} not found.")

# Prediction function
def predict(image: Image.Image, model, class_names) -> tuple[str, float]:
    try:
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            print(f"Probabilities: {probabilities}")
            confidence, predicted = torch.max(probabilities, 1)
            class_index = predicted.item()
        return class_names[class_index], confidence.item()
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown", 0.0

# Two-stage prediction
def classify_image(image: Image.Image) -> tuple[dict, dict]:
    produce_result = {'class': 'unknown', 'confidence': 0.0}
    variation_result = {'class': 'unknown', 'confidence': 0.0}

    if produce_model is None:
        print("Produce classifier not loaded.")
        return produce_result, variation_result

    # First stage: Predict produce type
    produce_class, produce_confidence = predict(image, produce_model, produce_class_names)
    produce_result = {'class': produce_class, 'confidence': produce_confidence}

    # Second stage: Predict variation (if applicable)
    variation_model = variation_models.get(produce_class)
    if variation_model:
        variation_class, variation_confidence = predict(image, variation_model, variation_class_names[produce_class])
        variation_result = {'class': variation_class, 'confidence': variation_confidence}
    else:
        print(f"No variation model for {produce_class}.")

    return produce_result, variation_result

# ============= RECIPE SEARCH SECTION =============

class RecipeBERTModel(nn.Module):
    """BERT-based model for recipe embedding"""
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=768, num_labels=256):
        super(RecipeBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        embeddings = self.projection(pooled_output)
        return embeddings

class RecipeSearchEngine:
    """Recipe search functionality using trained BERT model"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.recipe_embeddings = None
        self.recipe_ids = None
        self.recipes_df = None
        self.loaded = False
        
    def load_model(self, model_path='recipe_bert_model.pth'):
        """Load the trained recipe BERT model"""
        try:
            print("Loading recipe search model...")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"❌ Recipe model file {model_path} not found!")
                return False
            
            # Initialize tokenizer
            print("Loading BERT tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print("✅ BERT tokenizer loaded successfully!")
            
            # Load model checkpoint (using weights_only=False for compatibility with numpy arrays)
            print(f"Loading model checkpoint from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print("✅ Checkpoint loaded successfully!")
            
            # Initialize model
            print("Initializing BERT model...")
            self.model = RecipeBERTModel()
            self.model.to(self.device)
            
            # Load model weights
            print("Loading model state dict...")
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model weights loaded successfully!")
            else:
                print("❌ 'model_state_dict' not found in checkpoint!")
                return False
            
            # Load recipe embeddings
            if 'recipe_embeddings' in checkpoint:
                self.recipe_embeddings = checkpoint['recipe_embeddings']
                print(f"✅ Recipe embeddings loaded! Shape: {self.recipe_embeddings.shape}")
            else:
                print("❌ 'recipe_embeddings' not found in checkpoint!")
                return False
            
            # Load recipe IDs
            if 'recipe_ids' in checkpoint:
                self.recipe_ids = checkpoint['recipe_ids']
                print(f"✅ Recipe IDs loaded! Count: {len(self.recipe_ids)}")
            else:
                print("❌ 'recipe_ids' not found in checkpoint!")
                return False
            
            # Load recipe data
            if 'recipes_data' in checkpoint:
                print("Loading recipe data from checkpoint...")
                self.recipes_df = pd.DataFrame(checkpoint['recipes_data'])
                print(f"✅ Recipe data loaded from checkpoint! Count: {len(self.recipes_df)} recipes")
            else:
                # Try to load from CSV file
                csv_files = ['RAW_recipes.csv', 'recipes.csv']
                loaded_csv = False
                
                for csv_file in csv_files:
                    if os.path.exists(csv_file):
                        print(f"Loading recipe data from {csv_file}...")
                        self.recipes_df = pd.read_csv(csv_file)
                        print(f"✅ Recipe data loaded from {csv_file}! Count: {len(self.recipes_df)} recipes")
                        loaded_csv = True
                        break
                
                if not loaded_csv:
                    print("❌ No recipe data found! Checked for:")
                    for csv_file in csv_files:
                        print(f"  - {csv_file}")
                    print("Please ensure recipe data is either saved in the checkpoint or available as a CSV file.")
                    return False
            
            # Set model to evaluation mode
            self.model.eval()
            self.loaded = True
            
            print(f"✅ Recipe search engine fully loaded and ready!")
            print(f"   - Device: {self.device}")
            print(f"   - Recipe count: {len(self.recipes_df)}")
            print(f"   - Embedding shape: {self.recipe_embeddings.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading recipe model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_recipes(self, query_tags, top_k=5):
        """Search for recipes based on input tags using BERT embeddings"""
        if not self.loaded:
            print("❌ Recipe search engine not loaded!")
            return []
        
        if len(query_tags) < 5:
            print(f"❌ Need at least 5 tags, got {len(query_tags)}")
            return []
        
        try:
            print(f"🔍 Searching for recipes with tags: {query_tags}")
            
            # Create query text
            query_text = f"Tags: {' '.join(query_tags)}"
            print(f"Query text: {query_text}")
            
            # Tokenize query
            encoding = self.tokenizer(
                query_text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            # Generate query embedding
            print("Generating query embedding...")
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                query_embedding = self.model(input_ids, attention_mask).cpu().numpy()
            
            print(f"Query embedding shape: {query_embedding.shape}")
            
            # Compute similarities
            print("Computing similarities...")
            similarities = cosine_similarity(query_embedding, self.recipe_embeddings)[0]
            print(f"Similarities computed. Max: {similarities.max():.4f}, Min: {similarities.min():.4f}")
            
            # Get top-k recipes
            top_indices = np.argsort(similarities)[::-1][:top_k]
            print(f"Top {top_k} indices: {top_indices}")
            print(f"Top {top_k} scores: {similarities[top_indices]}")
            
            # Prepare results
            results = []
            for idx in top_indices:
                try:
                    recipe_id = self.recipe_ids[idx]
                    recipe = self.recipes_df[self.recipes_df['id'] == recipe_id].iloc[0]
                    
                    # Parse tags and ingredients safely
                    tags = recipe['tags']
                    if isinstance(tags, str):
                        try:
                            tags = ast.literal_eval(tags)
                        except:
                            tags = tags.split(',') if ',' in tags else [tags]
                    elif not isinstance(tags, list):
                        tags = []
                    
                    ingredients = recipe['ingredients']
                    if isinstance(ingredients, str):
                        try:
                            ingredients = ast.literal_eval(ingredients)
                        except:
                            ingredients = ingredients.split(',') if ',' in ingredients else [ingredients]
                    elif not isinstance(ingredients, list):
                        ingredients = []
                    
                    result = {
                        'id': int(recipe_id),
                        'name': recipe['name'],
                        'score': float(similarities[idx]),
                        'tags': tags[:10],  # Limit tags for display
                        'ingredients': ingredients[:10],  # Limit ingredients for display
                        'minutes': int(recipe['minutes']) if pd.notna(recipe['minutes']) else None,
                        'description': recipe['description'] if pd.notna(recipe['description']) else "",
                        'n_steps': int(recipe['n_steps']) if pd.notna(recipe['n_steps']) else 0
                    }
                    
                    results.append(result)
                    print(f"✅ Added recipe: {result['name']} (score: {result['score']:.4f})")
                    
                except Exception as e:
                    print(f"❌ Error processing recipe at index {idx}: {e}")
                    continue
            
            print(f"✅ Returning {len(results)} recipe results")
            return results
            
        except Exception as e:
            print(f"❌ Error searching recipes: {e}")
            import traceback
            traceback.print_exc()
            return []

# Initialize recipe search engine
recipe_search_engine = RecipeSearchEngine()

# Try to load the recipe model
print("Initializing recipe search engine...")
recipe_model_loaded = recipe_search_engine.load_model('recipe_bert_model.pth')

if recipe_model_loaded:
    print("✅ Recipe search engine initialized successfully!")
else:
    print("❌ Recipe search engine failed to initialize.")

# Recipe search function for external use
def search_recipes(tags, top_k=5):
    """Search for recipes based on tags"""
    if not recipe_model_loaded:
        return {
            'success': False,
            'error': 'Recipe search model not loaded',
            'results': []
        }
    
    if len(tags) < 5:
        return {
            'success': False,
            'error': f'Please provide at least 5 tags. You provided {len(tags)} tags.',
            'results': []
        }
    
    try:
        results = recipe_search_engine.search_recipes(tags, top_k)
        return {
            'success': True,
            'results': results
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'results': []
        }