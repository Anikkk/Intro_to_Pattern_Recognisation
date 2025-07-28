import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import json
import ast
from tqdm import tqdm
import pickle

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class RecipeDataset(Dataset):
    """Custom dataset for recipe data"""
    def __init__(self, recipes_df, tokenizer, max_length=512):
        self.recipes_df = recipes_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.recipes_df)
    
    def __getitem__(self, idx):
        recipe = self.recipes_df.iloc[idx]
        
        # Combine recipe information into a single text
        text = self._create_recipe_text(recipe)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'recipe_id': recipe['id'],
            'recipe_name': recipe['name']
        }
    
    def _create_recipe_text(self, recipe):
        """Create a comprehensive text representation of the recipe"""
        # Parse tags if they're in string format
        tags = recipe['tags']
        if isinstance(tags, str):
            try:
                tags = ast.literal_eval(tags)
            except:
                tags = []
        
        # Parse ingredients
        ingredients = recipe['ingredients']
        if isinstance(ingredients, str):
            try:
                ingredients = ast.literal_eval(ingredients)
            except:
                ingredients = []
        
        # Combine all text elements
        text_parts = [
            f"Recipe: {recipe['name']}",
            f"Tags: {' '.join(tags)}",
            f"Ingredients: {' '.join(ingredients)}",
            f"Description: {recipe['description']}" if pd.notna(recipe['description']) else "",
            f"Time: {recipe['minutes']} minutes" if pd.notna(recipe['minutes']) else ""
        ]
        
        return " ".join([part for part in text_parts if part])

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
    """Main class for recipe search functionality"""
    def __init__(self, model_path=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = RecipeBERTModel()
        self.model.to(device)
        
        if model_path:
            self.load_model(model_path)
        
        self.recipe_embeddings = None
        self.recipes_df = None
        
    def train_model(self, recipes_df, interactions_df, epochs=5, batch_size=16, learning_rate=2e-5):
        """Fine-tune BERT model on recipe data"""
        print("Preparing data for training...")
        
        # Create dataset
        dataset = RecipeDataset(recipes_df, self.tokenizer)
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training with contrastive learning
        print("Starting training...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                embeddings = self.model(input_ids, attention_mask)
                
                # Simple contrastive loss - recipes in same batch should have different embeddings
                loss = self._contrastive_loss(embeddings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        print("Training completed!")
        
    def _contrastive_loss(self, embeddings, temperature=0.07):
        """Compute contrastive loss for embeddings"""
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Create labels (each recipe is its own class)
        batch_size = embeddings.shape[0]
        labels = torch.arange(batch_size).to(device)
        
        # Compute cross entropy loss
        loss = nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def generate_embeddings(self, recipes_df):
        """Generate embeddings for all recipes"""
        print("Generating recipe embeddings...")
        self.recipes_df = recipes_df
        self.model.eval()
        
        dataset = RecipeDataset(recipes_df, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_embeddings = []
        recipe_ids = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                embeddings = self.model(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
                recipe_ids.extend(batch['recipe_id'].numpy())
        
        self.recipe_embeddings = np.vstack(all_embeddings)
        self.recipe_ids = np.array(recipe_ids)
        
        print(f"Generated embeddings for {len(self.recipe_embeddings)} recipes")
        
    def search(self, query_tags, top_k=10):
        """Search for recipes based on input tags"""
        if self.recipe_embeddings is None:
            raise ValueError("Recipe embeddings not generated. Call generate_embeddings first.")
        
        # Create query text
        query_text = f"Tags: {' '.join(query_tags)}"
        
        # Tokenize query
        encoding = self.tokenizer(
            query_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Generate query embedding
        self.model.eval()
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            query_embedding = self.model(input_ids, attention_mask).cpu().numpy()
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.recipe_embeddings)[0]
        
        # Get top-k recipes
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            recipe_id = self.recipe_ids[idx]
            recipe = self.recipes_df[self.recipes_df['id'] == recipe_id].iloc[0]
            
            results.append({
                'id': int(recipe_id),
                'name': recipe['name'],
                'score': float(similarities[idx]),
                'tags': ast.literal_eval(recipe['tags']) if isinstance(recipe['tags'], str) else recipe['tags'],
                'ingredients': ast.literal_eval(recipe['ingredients']) if isinstance(recipe['ingredients'], str) else recipe['ingredients'],
                'minutes': int(recipe['minutes']) if pd.notna(recipe['minutes']) else None,
                'description': recipe['description'] if pd.notna(recipe['description']) else ""
            })
        
        return results
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'recipe_embeddings': self.recipe_embeddings,
            'recipe_ids': self.recipe_ids
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.recipe_embeddings = checkpoint.get('recipe_embeddings')
        self.recipe_ids = checkpoint.get('recipe_ids')
        print(f"Model loaded from {path}")

# Main training and evaluation script
def main():
    # Load data
    print("Loading data...")
    recipes_df = pd.read_csv('RAW_recipes.csv')
    interactions_df = pd.read_csv('RAW_Interactions.csv')
    
    print(f"Loaded {len(recipes_df)} recipes and {len(interactions_df)} interactions")
    
    # Initialize search engine
    search_engine = RecipeSearchEngine()
    
    # Train model
    search_engine.train_model(recipes_df, interactions_df, epochs=3, batch_size=16)
    
    # Generate embeddings for all recipes
    search_engine.generate_embeddings(recipes_df)
    
    # Save model
    search_engine.save_model('recipe_bert_model.pth')
    
    # Test queries
    test_queries = [
        ['healthy', 'quick', 'vegetarian'],
        ['mexican', 'spicy', 'chicken'],
        ['dessert', 'chocolate', 'easy'],
        ['pasta', 'italian', 'cheese'],
        ['asian', 'noodles', 'vegetables'],
        ['breakfast', 'eggs', 'quick'],
        ['soup', 'winter', 'hearty'],
        ['salad', 'summer', 'fresh'],
        ['baking', 'bread', 'homemade'],
        ['seafood', 'grilled', 'lemon']
    ]
    
    print("\n" + "="*50)
    print("TESTING SEARCH ENGINE")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search_engine.search(query, top_k=5)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']} (Score: {result['score']:.3f})")
            print(f"   Tags: {', '.join(result['tags'][:5])}...")
            print(f"   Time: {result['minutes']} minutes")
    
    # Save results for analysis
    with open('search_results.json', 'w') as f:
        all_results = {}
        for query in test_queries:
            all_results[' '.join(query)] = search_engine.search(query, top_k=10)
        json.dump(all_results, f, indent=2)
    
    print("\nSearch results saved to 'search_results.json'")

if __name__ == "__main__":
    main()