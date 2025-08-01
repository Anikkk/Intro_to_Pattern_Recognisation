import streamlit as st
from PIL import Image
from backend import classify_image, search_recipes, recipe_search_engine
import json

st.set_page_config(
    page_title="Veggie Classifier & Recipe Search", 
    layout="wide",
    page_icon="🥦"
)

st.markdown("""
<style>
    .main-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e1e5e9;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #2c3e50 !important;
    }
    
    .recipe-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        color: #2c3e50 !important;
    }
    
    .recipe-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .recipe-card h3 {
        color: #2c3e50 !important;
        font-weight: bold;
    }
    
    .tag-pill {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 3px;
        display: inline-block;
        font-size: 12px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .ingredient-pill {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
        padding: 4px 10px;
        border-radius: 15px;
        margin: 2px;
        display: inline-block;
        font-size: 11px;
        font-weight: 500;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
        border: 1px solid #90caf9;
        color: #1565c0 !important;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
        color: #155724 !important;
        font-weight: 500;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fdf2f2 0%, #fed7d7 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #e53e3e;
        margin: 10px 0;
        color: #721c24 !important;
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e6f3ff 0%, #b3d9ff 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin: 10px 0;
        color: #004085 !important;
        font-weight: 500;
    }
    
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Ensure all text in main sections is visible */
    .main-section * {
        color: #2c3e50 !important;
    }
    
    .main-section h2, .main-section h3, .main-section h4 {
        color: #34495e !important;
        font-weight: bold !important;
    }
    
    /* Fix Streamlit metric labels */
    div[data-testid="metric-container"] > div > div > div {
        color: #2c3e50 !important;
    }
    
    /* Fix any remaining white text issues */
    .stMarkdown {
        color: #2c3e50 !important;
    }
    
    /* Recipe card text visibility */
    .recipe-card * {
        color: #2c3e50 !important;
    }
    
    .recipe-card strong {
        color: #2c3e50 !important;
        font-weight: bold !important;
    }
    
    /* Sidebar styling fixes */
    .css-1d391kg {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        color: #2c3e50 !important;
    }
    
    .css-1cypcdb {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        color: #2c3e50 !important;
    }
    
    /* Sidebar text styling */
    .css-1d391kg * {
        color: #2c3e50 !important;
    }
    
    .css-1cypcdb * {
        color: #2c3e50 !important;
    }
    
    /* Sidebar headers */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4 {
        color: #2c3e50 !important;
        font-weight: bold !important;
    }
    
    .css-1cypcdb h1, .css-1cypcdb h2, .css-1cypcdb h3, .css-1cypcdb h4 {
        color: #2c3e50 !important;
        font-weight: bold !important;
    }
    
    /* Sidebar markdown text */
    .css-1d391kg .stMarkdown {
        color: #2c3e50 !important;
    }
    
    .css-1cypcdb .stMarkdown {
        color: #2c3e50 !important;
    }
    
    /* Sidebar specific styling for better visibility */
    div[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
    }
    
    div[data-testid="stSidebar"] * {
        color: #2c3e50 !important;
    }
    
    div[data-testid="stSidebar"] h1, 
    div[data-testid="stSidebar"] h2, 
    div[data-testid="stSidebar"] h3, 
    div[data-testid="stSidebar"] h4 {
        color: #2c3e50 !important;
        font-weight: bold !important;
    }
    
    /* Sidebar code blocks and highlights */
    div[data-testid="stSidebar"] code {
        background-color: #e9ecef !important;
        color: #2c3e50 !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    
    /* Sidebar metrics */
    div[data-testid="stSidebar"] div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 5px 0 !important;
    }
    
    div[data-testid="stSidebar"] div[data-testid="metric-container"] * {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)


st.title("🥦🥜🌶️🥭 Veggie Classifier & Recipe Search")
st.markdown("---")


col1, col2 = st.columns([1, 1])



with col1:
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.markdown("## 📸 Image Classification")
    st.markdown("Upload an image of a **mango**, **pepper**, **nut**, or **broccoli**")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        col_classify, col_clear = st.columns([3, 1])
        
        with col_classify:
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Classifying..."):
                    produce_result, variation_result = classify_image(image)
                    
                    st.markdown("### 🎯 Classification Results")
                    if produce_result['class'] != 'unknown':
                        st.markdown(f"""
                        <div class="success-box">
                            <span style="color: #155724; font-weight: bold;">Type:</span> 
                            <span style="color: #155724;">{produce_result['class']}</span><br>
                            <span style="color: #155724; font-weight: bold;">Confidence:</span> 
                            <span style="color: #155724;">{produce_result['confidence']:.2%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if variation_result['class'] != 'unknown':
                            st.markdown(f"""
                            <div class="success-box">
                                <span style="color: #155724; font-weight: bold;">State:</span> 
                                <span style="color: #155724;">{variation_result['class']}</span><br>
                                <span style="color: #155724; font-weight: bold;">Confidence:</span> 
                                <span style="color: #155724;">{variation_result['confidence']:.2%}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.session_state['classified_produce'] = produce_result['class'].lower()
                        st.markdown(f"""
                        <div class="info-box">
                            <span style="color: #004085; font-weight: bold;">💡 Tip:</span> 
                            <span style="color: #004085;">Add '{produce_result['class'].lower()}' to your recipe search tags!</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="error-box">
                            <span style="color: #721c24; font-weight: bold;">❌ Could not classify image.</span>
                            <span style="color: #721c24;">Please try a clearer image.</span>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


with col2:
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.markdown("## 🍳 Recipe Search")
    st.markdown("**Minimum 5 tags required** for accurate recipe matching")
    
    if 'tags' not in st.session_state:
        st.session_state.tags = []
    
    # Add new tag section
    st.markdown("### ➕ Add Tags")
    tag_col1, tag_col2 = st.columns([3, 1])
    
    with tag_col1:
        new_tag = st.text_input("Enter a tag:", placeholder="e.g., healthy, quick, spicy", key="new_tag_input")
    
    with tag_col2:
        if st.button("Add", type="primary", use_container_width=True) and new_tag:
            if new_tag.lower().strip() not in st.session_state.tags:
                st.session_state.tags.append(new_tag.lower().strip())
                st.rerun()
            else:
                st.warning("Tag already added!")
    
    # Quick tags section
    st.markdown("### 🚀 Quick Add Tags")
    quick_tags = [
        ['healthy', 'quick', 'easy'],
        ['vegetarian', 'spicy', 'dessert'],
        ['italian', 'mexican', 'asian'],
        ['chicken', 'beef', 'seafood']
    ]
    
    for row in quick_tags:
        cols = st.columns(3)
        for i, tag in enumerate(row):
            with cols[i]:
                if st.button(tag.title(), key=f"quick_{tag}", use_container_width=True):
                    if tag not in st.session_state.tags:
                        st.session_state.tags.append(tag)
                        st.rerun()

    if 'classified_produce' in st.session_state:
        if st.button(f"➕ Add Classified: '{st.session_state.classified_produce.title()}'", 
                     type="secondary", use_container_width=True):
            if st.session_state.classified_produce not in st.session_state.tags:
                st.session_state.tags.append(st.session_state.classified_produce)
                st.rerun()
    

    st.markdown("### 🏷️ Current Tags")
    if st.session_state.tags:

        tag_html = ""
        for tag in st.session_state.tags:
            tag_html += f'<span class="tag-pill">{tag}</span> '
        st.markdown(tag_html, unsafe_allow_html=True)
        
        st.markdown(f"**Total:** {len(st.session_state.tags)} tags")
        
        if len(st.session_state.tags) > 0:
            col_remove1, col_remove2 = st.columns([3, 1])
            with col_remove1:
                tag_to_remove = st.selectbox("Remove tag:", [""] + st.session_state.tags, key="remove_select")
            with col_remove2:
                if st.button("Remove", use_container_width=True) and tag_to_remove:
                    st.session_state.tags.remove(tag_to_remove)
                    st.rerun()
        
        if st.button("🗑️ Clear All Tags", use_container_width=True):
            st.session_state.tags = []
            st.rerun()
    else:
        st.markdown("""
        <div class="info-box">
            <span style="color: #004085; font-weight: bold;">📝 No tags added yet.</span>
            <span style="color: #004085;">Add at least 5 tags to search for recipes.</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Search button
    remaining_tags = max(0, 5 - len(st.session_state.tags))
    
    if len(st.session_state.tags) >= 5:
        if st.button("🔍 Search Recipes", type="primary", use_container_width=True):
            with st.spinner("Searching for recipes..."):
                result = search_recipes(st.session_state.tags, top_k=5)
                st.session_state['search_results'] = result
    else:
        st.button(f"🔍 Need {remaining_tags} more tags", disabled=True, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if 'search_results' in st.session_state:
    st.markdown("---")
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.markdown("## 📋 Recipe Search Results")
    
    results = st.session_state['search_results']
    
    if results['success']:
        recipes = results['results']
        
        if recipes:
            st.markdown(f"""
            <div class="success-box">
                <span style="color: #155724; font-weight: bold;">🎉 Found {len(recipes)} recipes!</span>
                <span style="color: #155724;">Sorted by relevance score.</span>
            </div>
            """, unsafe_allow_html=True)
            
            for i, recipe in enumerate(recipes, 1):
                st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                
                st.markdown(f"### #{i} {recipe['name']}")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("🎯 Relevance", f"{recipe['score']:.3f}")
                with metric_col2:
                    cook_time = f"{recipe['minutes']} min" if recipe['minutes'] else "N/A"
                    st.metric("⏱️ Cook Time", cook_time)
                with metric_col3:
                    st.metric("📝 Steps", recipe['n_steps'])
                with metric_col4:
                    ing_count = len(recipe['ingredients']) if recipe.get('ingredients') else 0
                    st.metric("🥘 Ingredients", ing_count)
                
                if recipe.get('tags'):
                    st.markdown("**🏷️ Tags:**")
                    tags_to_show = recipe['tags'][:10] if isinstance(recipe['tags'], list) else []
                    if tags_to_show:
                        tag_html = ""
                        for tag in tags_to_show:
                            tag_html += f'<span class="tag-pill">{tag}</span> '
                        st.markdown(tag_html, unsafe_allow_html=True)
                
 
                if recipe.get('ingredients'):
                    st.markdown("**🥘 Key Ingredients:**")
                    ingredients = recipe['ingredients'][:12] if isinstance(recipe['ingredients'], list) else []
                    if ingredients:
                        ingredient_html = ""
                        for ingredient in ingredients:
                            ingredient_html += f'<span class="ingredient-pill">{ingredient}</span> '
                        st.markdown(ingredient_html, unsafe_allow_html=True)
                

                if recipe.get('description') and recipe['description'].strip():
                    st.markdown("**📄 Description:**")
                    desc = recipe['description']
                    if len(desc) > 300:
                        desc = desc[:300] + "..."
                    st.markdown(f"*{desc}*")
                
 
                st.markdown(f"**🔗 Recipe ID:** `{recipe['id']}`")
                
                st.markdown('</div>', unsafe_allow_html=True)
                

                if i < len(recipes):
                    st.markdown("---")
        else:
            st.markdown("""
            <div class="error-box">
                <span style="color: #721c24; font-weight: bold;">😔 No recipes found!</span><br>
                <span style="color: #721c24;">Try adjusting your tags or adding more specific ingredients.</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        error_msg = results.get('error', 'Unknown error occurred')
        st.markdown(f"""
        <div class="error-box">
            <span style="color: #721c24; font-weight: bold;">❌ Search Failed:</span><br>
            <span style="color: #721c24;">{error_msg}</span>
        </div>
        """, unsafe_allow_html=True)
        

        st.markdown("""
        <div class="info-box">
            <span style="color: #004085; font-weight: bold;">💡 Troubleshooting:</span><br>
            <span style="color: #004085;">• Make sure you have trained and saved the model (.pth file)<br>
            • Ensure you have at least 5 tags<br>
            • Check that the model file contains recipe data<br>
            • Try restarting the application</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## ℹ️ How to Use")
    st.markdown("""
    **🚀 Quick Start:**
    1. **Upload** a vegetable image (optional)
    2. **Add** at least 5 recipe tags
    3. **Search** for matching recipes
    4. **Browse** top 5 results
    """)
    
    st.markdown("**💡 Tag Examples:**")
    

    st.markdown("""
    <div style="margin: 10px 0;">
        <strong style="color: #2c3e50;">🥘 Ingredients:</strong><br>
        <span style="background: #28a745; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">chicken</span>
        <span style="background: #28a745; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">broccoli</span>
        <span style="background: #28a745; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">beef</span>
    </div>
    
    <div style="margin: 10px 0;">
        <strong style="color: #2c3e50;">🍳 Cooking Style:</strong><br>
        <span style="background: #007bff; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">grilled</span>
        <span style="background: #007bff; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">baked</span>
        <span style="background: #007bff; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">stir-fry</span>
    </div>
    
    <div style="margin: 10px 0;">
        <strong style="color: #2c3e50;">🥗 Dietary:</strong><br>
        <span style="background: #17a2b8; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">healthy</span>
        <span style="background: #17a2b8; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">vegetarian</span>
        <span style="background: #17a2b8; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">low-carb</span>
    </div>
    
    <div style="margin: 10px 0;">
        <strong style="color: #2c3e50;">🌍 Cuisine:</strong><br>
        <span style="background: #6f42c1; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">italian</span>
        <span style="background: #6f42c1; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">mexican</span>
        <span style="background: #6f42c1; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">asian</span>
    </div>
    
    <div style="margin: 10px 0;">
        <strong style="color: #2c3e50;">⏱️ Time:</strong><br>
        <span style="background: #fd7e14; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">easy</span>
        <span style="background: #fd7e14; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">quick</span>
        <span style="background: #fd7e14; color: white; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">30-minutes</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📊 System Status")
    

    from backend import produce_model
    if produce_model is not None:
        st.success("✅ Image Classification Ready")
    else:
        st.error("❌ Image models not found")
    

    if hasattr(recipe_search_engine, 'loaded') and recipe_search_engine.loaded:
        st.success("✅ Recipe Search Ready")
        if hasattr(recipe_search_engine, 'recipes_data') and recipe_search_engine.recipes_data:
            st.info(f"📊 {len(recipe_search_engine.recipes_data):,} recipes loaded")
        if hasattr(recipe_search_engine, 'recipe_embeddings') and recipe_search_engine.recipe_embeddings is not None:
            st.info(f"🧠 Embeddings: {recipe_search_engine.recipe_embeddings.shape}")
    else:
        st.error("❌ Recipe search not loaded")
        st.caption("Check if .pth model file exists")
    
    # Show current session stats
    if 'tags' in st.session_state:
        st.metric("🏷️ Active Tags", len(st.session_state.tags))
        if st.session_state.tags:
            with st.expander("View Tags"):
                for tag in st.session_state.tags:
                    st.text(f"• {tag}")
    
    if 'search_results' in st.session_state:
        if st.session_state['search_results']['success']:
            st.metric("🍳 Recipes Found", len(st.session_state['search_results']['results']))
    

    st.markdown("---")
    st.markdown("## 🤖 Model Info")
    st.caption("**Image:** ResNet-18 based classifier")
    st.caption("**Recipe:** DistilBERT embeddings")
    st.caption("**Search:** Cosine similarity matching")
    

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px;">
    <h4>🍳 Recipe Search Engine</h4>
    <p>Powered by DistilBERT embeddings for intelligent recipe matching</p>
    <small>Add at least 5 tags for the best results • Supports 200,000+ recipes</small>
</div>
""", unsafe_allow_html=True)
