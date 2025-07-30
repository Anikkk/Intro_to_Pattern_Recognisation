import streamlit as st
from PIL import Image
from backend import classify_image, search_recipes, recipe_search_engine
import json

# Page configuration
st.set_page_config(
    page_title="Veggie Classifier & Recipe Search", 
    layout="wide",
    page_icon="🥦"
)

# Simple and clean CSS
st.markdown("""
<style>
    .main-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin-bottom: 15px;
    }
    
    .recipe-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .tag-pill {
        background-color: #007bff;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        margin: 2px;
        display: inline-block;
        font-size: 12px;
    }
    
    .metric-box {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🥦🥜🌶️🥭 Veggie Classifier & Recipe Search")
st.markdown("---")

# Create two columns
col1, col2 = st.columns([1, 1])

# ============= IMAGE CLASSIFICATION =============
with col1:
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.header("📸 Image Classification")
    st.write("Upload an image of a **mango**, **pepper**, **nut**, or **broccoli**")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner("Classifying..."):
                produce_result, variation_result = classify_image(image)
                
                st.subheader("🎯 Results")
                if produce_result['class'] != 'unknown':
                    st.success(f"**Type:** {produce_result['class']} (Confidence: {produce_result['confidence']:.2%})")
                    
                    if variation_result['class'] != 'unknown':
                        st.success(f"**State:** {variation_result['class']} (Confidence: {variation_result['confidence']:.2%})")
                    
                    # Store for recipe search
                    st.session_state['classified_produce'] = produce_result['class'].lower()
                    st.info(f"💡 Add '{produce_result['class'].lower()}' to recipe search!")
                else:
                    st.error("Could not classify image")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============= RECIPE SEARCH =============
with col2:
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.header("🍳 Recipe Search")
    st.write("Add at least 5 tags to search for recipes")
    
    # Initialize tags
    if 'tags' not in st.session_state:
        st.session_state.tags = []
    
    # Add new tag
    new_tag = st.text_input("Enter a tag:", placeholder="e.g., healthy, quick")
    if st.button("➕ Add Tag") and new_tag:
        if new_tag.lower() not in st.session_state.tags:
            st.session_state.tags.append(new_tag.lower())
            st.rerun()
    
    # Quick tags
    st.write("**Quick Add:**")
    quick_cols = st.columns(3)
    quick_tags = ['healthy', 'quick', 'easy', 'vegetarian', 'spicy', 'dessert']
    
    for i, tag in enumerate(quick_tags):
        with quick_cols[i % 3]:
            if st.button(tag.title(), key=f"quick_{tag}"):
                if tag not in st.session_state.tags:
                    st.session_state.tags.append(tag)
                    st.rerun()
    
    # Add classified produce
    if 'classified_produce' in st.session_state:
        if st.button(f"➕ Add '{st.session_state.classified_produce}'"):
            if st.session_state.classified_produce not in st.session_state.tags:
                st.session_state.tags.append(st.session_state.classified_produce)
                st.rerun()
    
    # Show current tags
    st.write("**Current Tags:**")
    if st.session_state.tags:
        # Display tags
        tag_cols = st.columns(3)
        for i, tag in enumerate(st.session_state.tags):
            with tag_cols[i % 3]:
                if st.button(f"❌ {tag}", key=f"remove_{tag}_{i}"):
                    st.session_state.tags.remove(tag)
                    st.rerun()
        
        st.write(f"Total: {len(st.session_state.tags)} tags")
        
        if st.button("🗑️ Clear All"):
            st.session_state.tags = []
            st.rerun()
    else:
        st.info("No tags yet - add at least 5")
    
    # Search button
    if len(st.session_state.tags) >= 5:
        if st.button("🔍 Search Recipes", type="primary"):
            with st.spinner("Searching..."):
                result = search_recipes(st.session_state.tags)
                st.session_state['search_results'] = result
    else:
        remaining = 5 - len(st.session_state.tags)
        st.button(f"Need {remaining} more tags", disabled=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============= RECIPE RESULTS =============
if 'search_results' in st.session_state:
    st.markdown("---")
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.header("📋 Recipe Results")
    
    results = st.session_state['search_results']
    
    if results['success']:
        recipes = results['results']
        
        if recipes:
            st.success(f"Found {len(recipes)} recipes!")
            
            for i, recipe in enumerate(recipes, 1):
                st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                
                # Recipe header
                st.subheader(f"{i}. {recipe['name']}")
                
                # Metrics in columns
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("BERT Score", f"{recipe['score']:.3f}")
                with metric_col2:
                    cook_time = f"{recipe['minutes']} min" if recipe['minutes'] else "N/A"
                    st.metric("Cook Time", cook_time)
                with metric_col3:
                    st.metric("Steps", recipe['n_steps'])
                
                # Tags
                if recipe.get('tags'):
                    st.write("**Tags:**")
                    tags_to_show = recipe['tags'][:8] if isinstance(recipe['tags'], list) else []
                    if tags_to_show:
                        tag_html = ""
                        for tag in tags_to_show:
                            tag_html += f'<span class="tag-pill">{tag}</span> '
                        st.markdown(tag_html, unsafe_allow_html=True)
                
                # Ingredients
                if recipe.get('ingredients'):
                    st.write("**Ingredients:**")
                    ingredients = recipe['ingredients'][:10] if isinstance(recipe['ingredients'], list) else []
                    if ingredients:
                        ingredients_text = ", ".join(ingredients)
                        st.write(ingredients_text)
                
                # Description
                if recipe.get('description') and recipe['description'].strip():
                    st.write("**Description:**")
                    desc = recipe['description']
                    if len(desc) > 200:
                        desc = desc[:200] + "..."
                    st.write(desc)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.warning("No recipes found. Try different tags!")
    else:
        st.error(f"Search failed: {results['error']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============= SIDEBAR =============
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    **Steps:**
    1. Upload produce image
    2. Classify the image  
    3. Add 5+ tags for recipes
    4. Search for recipes
    5. Browse results
    """)
    
    st.markdown("---")
    st.header("📊 Status")
    
    # Check systems
    from backend import produce_model
    if produce_model is not None:
        st.success("✅ Image Classification Ready")
    else:
        st.error("❌ Image models not found")
    
    if hasattr(recipe_search_engine, 'loaded') and recipe_search_engine.loaded:
        st.success("✅ Recipe Search Ready")
        if hasattr(recipe_search_engine, 'recipes_df'):
            st.info(f"📊 {len(recipe_search_engine.recipes_df)} recipes")
    else:
        st.error("❌ Recipe search not loaded")
    
    if 'search_results' in st.session_state:
        if st.session_state['search_results']['success']:
            st.metric("Recipes Found", len(st.session_state['search_results']['results']))
    
    if 'tags' in st.session_state:
        st.metric("Active Tags", len(st.session_state.tags))