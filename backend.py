import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import ast
import json

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


class RecipeDistilBERTModel(nn.Module):
    """DistilBERT-based model for recipe embedding - Updated to match training code"""
    def __init__(self, model_name='distilbert-base-uncased', hidden_size=768, embedding_dim=256):
        super(RecipeDistilBERTModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(hidden_size, embedding_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
    
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        embeddings = self.projection(cls_output)
        return embeddings

class RecipeSearchEngine:
    """Recipe search functionality using trained DistilBERT model - NO CSV DEPENDENCY"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.recipe_embeddings = None
        self.recipe_ids = None
        self.recipes_data = None  # Store recipe data directly from .pth file
        self.loaded = False
        
    def load_model(self, model_path='recipe_distilbert_model.pth'):
        """Load the trained recipe DistilBERT model - NO CSV LOADING"""
        try:
            print("Loading recipe search model...")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print("Make sure you have trained and saved the model first!")
                return False
            
            # Initialize tokenizer (DistilBERT)
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            
            # Initialize DistilBERT model
            self.model = RecipeDistilBERTModel()
            self.model.to(self.device)
            
    
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                return False
            
            # Load recipe embeddings
            if 'recipe_embeddings' in checkpoint:
                self.recipe_embeddings = checkpoint['recipe_embeddings']
                print(f"✅ Recipe embeddings loaded! Shape: {self.recipe_embeddings.shape}")
            else:
                print("'recipe_embeddings' not found in checkpoint!")
                return False
            
            # Load recipe IDs
            if 'recipe_ids' in checkpoint:
                self.recipe_ids = checkpoint['recipe_ids']
                print(f" Recipe IDs loaded! Count: {len(self.recipe_ids)}")
            else:
                return False
            
            # Load recipe data - NO CSV LOADING, EVERYTHING FROM .PTH FILE
            if 'recipes_data' in checkpoint:
                print("Loading recipe data from checkpoint...")
                self.recipes_data = checkpoint['recipes_data']
                print(f"✅ Recipe data loaded from checkpoint! Count: {len(self.recipes_data)} recipes")
            else:
                return False
            
            # Set model to evaluation mode
            self.model.eval()
            self.loaded = True
            print(f"   - Device: {self.device}")
            print(f"   - Recipe count: {len(self.recipes_data)}")
            print(f"   - Embedding shape: {self.recipe_embeddings.shape}")
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False
    
    def search_recipes(self, query_tags, top_k=5):
        if not self.loaded:
            return []
        
        if len(query_tags) < 5:
            return []
        
        try:
            print(f"🔍 Searching for recipes with tags: {query_tags}")
            
            # Create query text (same format as training)
            query_text = f"Tags: {' '.join(query_tags)}"
            print(f"Query text: {query_text}")
            
            # Tokenize query (using DistilBERT tokenizer)
            encoding = self.tokenizer(
                query_text,
                truncation=True,
                padding='max_length',
                max_length=128,  
                return_tensors='pt'
            )
            
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                query_embedding = self.model(input_ids, attention_mask).cpu().numpy()
            
            print(f"Query embedding shape: {query_embedding.shape}")
            
            # Compute similarities
            similarities = cosine_similarity(query_embedding, self.recipe_embeddings)[0]
            
            top_indices = np.argsort(similarities)[::-1][:top_k]


            results = []
            for idx in top_indices:
                try:
                    recipe_id = self.recipe_ids[idx]
                    recipe = None
                    for r in self.recipes_data:
                        if r['id'] == recipe_id:
                            recipe = r
                            break
                    
                    if recipe is None:
                        print(f"❌ Recipe with ID {recipe_id} not found in stored data")
                        continue
                    
                    tags = recipe.get('parsed_tags', [])
                    if not tags:
                        tags_raw = recipe.get('tags', [])
                        if isinstance(tags_raw, str):
                            try:
                                tags = ast.literal_eval(tags_raw)
                            except:
                                tags = tags_raw.split(',') if ',' in tags_raw else [tags_raw]
                        elif isinstance(tags_raw, list):
                            tags = tags_raw
                    
                    ingredients = recipe.get('parsed_ingredients', [])
                    if not ingredients:
                        ingredients_raw = recipe.get('ingredients', [])
                        if isinstance(ingredients_raw, str):
                            try:
                                ingredients = ast.literal_eval(ingredients_raw)
                            except:
                                ingredients = ingredients_raw.split(',') if ',' in ingredients_raw else [ingredients_raw]
                        elif isinstance(ingredients_raw, list):
                            ingredients = ingredients_raw
                    
                    result = {
                        'id': int(recipe_id),
                        'name': recipe.get('name', 'Unknown Recipe'),
                        'score': float(similarities[idx]),
                        'tags': tags[:10],  # Limit tags for display
                        'ingredients': ingredients[:10],  # Limit ingredients for display
                        'minutes': int(recipe.get('minutes', 0)) if recipe.get('minutes') else None,
                        'description': recipe.get('description', ''),
                        'n_steps': int(recipe.get('n_steps', 0)) if recipe.get('n_steps') else 0
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    continue
            return results
            
        except Exception as e:
            print(f"❌ Error searching recipes: {e}")
            import traceback
            traceback.print_exc()
            return []

recipe_search_engine = RecipeSearchEngine()

model_files = ['recipe_distilbert_model.pth']
recipe_model_loaded = False

for model_file in model_files:
    if os.path.exists(model_file):
        print(f"Found model file: {model_file}")
        recipe_model_loaded = recipe_search_engine.load_model(model_file)
        if recipe_model_loaded:
            break

if recipe_model_loaded:
    print("✅ Recipe search engine initialized successfully!")
else:
    print("❌ Recipe search engine failed to initialize.")
    for f in os.listdir('.'):
        if f.endswith('.pth'):
            print(f"  - {f}")


def search_recipes(tags, top_k=5):
    """Search for recipes based on tags"""
    if not recipe_model_loaded:
        return {
            'success': False,
            'error': 'Recipe search model not loaded. Make sure the .pth file exists and contains recipe data.',
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
