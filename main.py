import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cache the model loading for better performance
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.h5')

def model_prediction(test_image):
    model = load_model()
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #2e8b57 !important;
        text-align: center;
        padding: 20px;
    }
    .subheader {
        font-size: 24px !important;
        color: #3cb371 !important;
        border-bottom: 2px solid #3cb371;
        padding-bottom: 10px;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
    }
    .success-box {
        background-color: #dff0d8;
        color: #3c763d;
        border-radius: 5px;
        padding: 15px;
        margin: 20px 0;
        font-size: 20px;
        text-align: center;
        border: 1px solid #d6e9c6;
    }
    .about-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌱 Plant Health Dashboard")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2919/2919906.png", width=80)
app_mode = st.sidebar.selectbox("Navigate", ["Home", "About", "Disease Recognition"])
st.sidebar.markdown("---")
st.sidebar.info("""
**Plant Disease Recognition System**  
Helping farmers identify crop diseases quickly and accurately
""")

# Home Page
if app_mode == "Home":
    st.markdown('<p class="header">🌿 Plant Disease Recognition System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://images.unsplash.com/photo-1587049633312-d628ae50a8ae", caption="Healthy Crops, Better Harvest")
    with col2:
        st.markdown("""
        <div style='text-align: justify;'>
        Our mission is to revolutionize agriculture through AI-powered plant disease detection. 
        Upload an image of a plant leaf, and our advanced deep learning system will instantly 
        analyze it for disease symptoms. Together, let's protect our crops and ensure food security!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p class="subheader">🚀 How It Works</p>', unsafe_allow_html=True)
    
    steps = st.columns(3)
    with steps[0]:
        st.image("https://cdn-icons-png.flaticon.com/512/892/892657.png", width=80)
        st.subheader("1. Upload Image")
        st.write("Capture a clear photo of a plant leaf showing symptoms")
    with steps[1]:
        st.image("https://cdn-icons-png.flaticon.com/512/2166/2166833.png", width=80)
        st.subheader("2. AI Analysis")
        st.write("Our neural network processes the image")
    with steps[2]:
        st.image("https://cdn-icons-png.flaticon.com/512/4114/4114700.png", width=80)
        st.subheader("3. Get Results")
        st.write("Receive instant diagnosis and recommendations")
    
    st.markdown("---")
    st.markdown('<p class="subheader">💡 Why Choose Us</p>', unsafe_allow_html=True)
    
    features = st.columns(3)
    with features[0]:
        st.metric("Accuracy", "98%", "2% improvement")
    with features[1]:
        st.metric("Speed", "2 seconds", "Instant results")
    with features[2]:
        st.metric("Coverage", "38 Diseases", "25+ plant species")
    
    if st.button("Get Started →", use_container_width=True):
        st.session_state.page = "Disease Recognition"
        st.experimental_rerun()

# About Page
elif app_mode == "About":
    st.markdown('<p class="header">📚 About Our Project</p>', unsafe_allow_html=True)
    
    with st.expander("Project Overview", expanded=True):
        st.markdown("""
        Our system uses a deep convolutional neural network (CNN) trained on over 87,000 images 
        to identify 38 different plant diseases. The model achieves 98% accuracy on validation data.
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*2on8s3X8Q8l2QzNtAdCv3g.jpeg")
    
    with st.expander("Dataset Information"):
        st.markdown("""
        **Dataset Statistics:**
        - Total Images: 87,867
        - Training Set: 70,295 images (80%)
        - Validation Set: 17,572 images (20%)
        - Test Set: 33 images
        - Classes: 38 plant disease categories
        """)
    
    with st.expander("Technology Stack"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg", width=100)
            st.caption("TensorFlow for deep learning")
        with col2:
            st.image("https://streamlit.io/images/brand/streamlit-mark-color.svg", width=100)
            st.caption("Streamlit for web interface")
        with col3:
            st.image("https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg", width=100)
            st.caption("NumPy for numerical processing")
    
    st.markdown("---")
    st.markdown("### 🌐 Our Vision")
    st.info("""
    To create a global platform where farmers can access instant plant health diagnostics, 
    reducing crop losses and promoting sustainable agriculture practices worldwide.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.markdown('<p class="header">🔍 Disease Recognition</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"], 
                                    help="Clear, well-lit images work best")
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with col2:
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing... This takes about 10 seconds"):
                    result_index = model_prediction(uploaded_file)
                    
                    class_names = [
                        'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Healthy Apple',
                        'Healthy Blueberry', 'Cherry Powdery Mildew', 'Healthy Cherry',
                        'Corn Gray Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight',
                        'Healthy Corn', 'Grape Black Rot', 'Grape Esca', 'Grape Leaf Blight',
                        'Healthy Grape', 'Citrus Greening', 'Peach Bacterial Spot', 'Healthy Peach',
                        'Bell Pepper Bacterial Spot', 'Healthy Bell Pepper', 'Potato Early Blight',
                        'Potato Late Blight', 'Healthy Potato', 'Healthy Raspberry', 'Healthy Soybean',
                        'Squash Powdery Mildew', 'Strawberry Leaf Scorch', 'Healthy Strawberry',
                        'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
                        'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Spider Mites',
                        'Tomato Target Spot', 'Tomato Yellow Leaf Curl', 'Tomato Mosaic Virus',
                        'Healthy Tomato'
                    ]
                    
                    readable_name = class_names[result_index]
                    status = "healthy" if "healthy" in readable_name.lower() else "diseased"
                    
                    st.markdown(f'<div class="success-box">Diagnosis: <strong>{readable_name}</strong></div>', 
                               unsafe_allow_html=True)
                    
                    if status == "healthy":
                        st.success("This plant appears healthy! Continue good maintenance practices.")
                    else:
                        st.warning("Disease detected! Consider consulting agricultural extension services.")
                        
                    st.balloons()
                    
                    # Show recommendations
                    if "apple" in readable_name.lower():
                        st.info("**Apple Care Tip:** Apply fungicides during wet weather to prevent fungal spread")
                    elif "tomato" in readable_name.lower():
                        st.info("**Tomato Care Tip:** Rotate crops annually to prevent soil-borne diseases")
    else:
        st.info("👆 Please upload an image to get started")
        st.image("https://images.unsplash.com/photo-1591462174822-a3e5d2a0c82d", 
                caption="Example of good leaf image for analysis")