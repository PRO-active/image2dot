import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from streamlit_cropper import st_cropper
from PIL import Image

def process_image(img, size, num_colors, contrast, offset, c_min, c_max, outline_color):
    # Resize image
    img_small = cv2.resize(img, (size, size))

    # Convert to RGB if it's RGBA
    if img_small.shape[2] == 4:
        img_small = cv2.cvtColor(img_small, cv2.COLOR_RGBA2RGB)

    # Create a mask for non-white pixels (assuming white is the background)
    is_visible = np.any(img_small != [255, 255, 255], axis=-1)

    # Reduce color palette
    visible_pixels = img_small[is_visible]
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(visible_pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.predict(img_small.reshape(-1, 3))

    img_palette = colors[labels].reshape(img_small.shape).astype(np.uint8)

    # Adjust contrast
    img_contrast = np.clip((img_palette.astype(int) - 128) * contrast + 128 + offset, 0, 255).astype(np.uint8)

    # Detect edges
    img_gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    img_canny = cv2.Canny(img_gray, c_min, c_max)

    # Combine results
    img_result = img_contrast.copy()
    img_result[img_canny > 0] = outline_color[:3]  # Use only RGB values of outline_color

    return img_result

st.title("ドット絵変換アプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Cropping
    st.write("画像をトリミングしてください")
    cropped_img = st_cropper(image, realtime_update=True, box_color="red", aspect_ratio=None)
    
    # Convert cropped image to numpy array
    img_array = np.array(cropped_img)

    # Parameters
    size = st.slider("出力サイズ", 16, 128, 64)
    num_colors = st.slider("色数", 2, 32, 8)
    contrast = st.slider("コントラスト", 0.5, 2.0, 1.0)
    offset = st.slider("明るさ調整", -50, 50, 0)
    c_min = st.slider("エッジ検出 (最小)", 0, 255, 100)
    c_max = st.slider("エッジ検出 (最大)", 0, 255, 200)
    outline_color = st.color_picker("輪郭線の色", "#000000")

    if st.button("変換開始"):
        outline_color_rgb = tuple(int(outline_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        result = process_image(img_array, size, num_colors, contrast, offset, c_min, c_max, outline_color_rgb)
        
        st.image(result, caption="変換後の画像", use_column_width=True)
        
        # Save button
        result_pil = Image.fromarray(result)
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="画像を保存",
            data=byte_im,
            file_name="dot_art.png",
            mime="image/png"
        )

st.write("注意: このアプリは画像処理に時間がかかる場合があります。しばらくお待ちください。")