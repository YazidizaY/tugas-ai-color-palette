import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

st.set_page_config(page_title="AI Color Palette Extractor", layout="centered")

st.title("🎨 AI Color Palette Generator")
st.write("Upload gambar kamu, dan AI (K-Means Clustering) akan mendeteksi 5 warna paling dominan secara otomatis.")

uploaded_file = st.file_uploader("Pilih gambar (PNG, JPG, JPEG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Gambar yang kamu upload", use_container_width=True)
    
    st.write("🔄 *Sedang menghitung warna dominan dengan Machine Learning...*")
    
    img_np = np.array(image.convert("RGB"))
    
    pixels = img_np.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    st.success("✨ Berhasil mengekstrak 5 warna dominan!")
    
    st.subheader("Palet Warna Dominan:")
    cols = st.columns(5)
    
    for i, color in enumerate(dominant_colors):
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        
        with cols[i]:
            st.markdown(
                f'<div style="background-color: {hex_color}; height: 100px; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0,0,0,0.15);"></div>',
                unsafe_allow_html=True
            )
            st.code(hex_color, language="text")
            st.caption(f"RGB: {color[0]},{color[1]},{color[2]}")
