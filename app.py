import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from streamlit_cropper import st_cropper
from PIL import Image, ExifTags, ImageEnhance
from io import BytesIO
import colorsys

def fix_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def quantize_colors(img, n_colors):
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    labels = kmeans.fit_predict(pixels)
    palette = kmeans.cluster_centers_
    quantized = palette[labels].reshape(img.shape).astype(np.uint8)
    return quantized

def apply_dithering(img, palette):
    h, w = img.shape[:2]
    dithered = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x].astype(int)
            new_pixel = palette[np.argmin(np.sum((palette - old_pixel) ** 2, axis=1))]
            dithered[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            if x + 1 < w:
                img[y, x + 1] = np.clip(img[y, x + 1] + quant_error * 7 / 16, 0, 255)
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] = np.clip(img[y + 1, x - 1] + quant_error * 3 / 16, 0, 255)
                img[y + 1, x] = np.clip(img[y + 1, x] + quant_error * 5 / 16, 0, 255)
                if x + 1 < w:
                    img[y + 1, x + 1] = np.clip(img[y + 1, x + 1] + quant_error * 1 / 16, 0, 255)
    return dithered

def adjust_saturation(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def process_image(img, size, num_colors, contrast, offset, c_min, c_max, outline_color, saturation, dither):
    # Resize image
    img_small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # Convert to RGB if it's RGBA
    if img_small.shape[2] == 4:
        img_small = cv2.cvtColor(img_small, cv2.COLOR_RGBA2RGB)

    # Adjust contrast and brightness
    img_adjusted = np.clip((img_small.astype(int) - 128) * contrast + 128 + offset, 0, 255).astype(np.uint8)

    # Adjust saturation
    img_saturated = adjust_saturation(img_adjusted, saturation)

    # Quantize colors
    palette = KMeans(n_clusters=num_colors, random_state=42).fit(img_saturated.reshape(-1, 3)).cluster_centers_
    
    if dither:
        img_quantized = apply_dithering(img_saturated, palette)
    else:
        img_quantized = quantize_colors(img_saturated, num_colors)

    # Detect edges
    img_gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    img_canny = cv2.Canny(img_gray, c_min, c_max)

    # Combine results
    img_result = img_quantized.copy()
    img_result[img_canny > 0] = outline_color[:3]

    return img_result

st.title("高精度ドット絵変換アプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = fix_image_orientation(image)
    
    st.write("画像をトリミングしてください")
    cropped_img = st_cropper(image, realtime_update=True, box_color="red", aspect_ratio=None)
    
    img_array = np.array(cropped_img)

    size = st.slider("出力サイズ", 16, 256, 64)
    num_colors = st.slider("色数", 2, 64, 16)
    contrast = st.slider("コントラスト", 0.5, 2.0, 1.0)
    offset = st.slider("明るさ調整", -50, 50, 0)
    c_min = st.slider("エッジ検出 (最小)", 0, 255, 100)
    c_max = st.slider("エッジ検出 (最大)", 0, 255, 200)
    outline_color = st.color_picker("輪郭線の色", "#000000")
    saturation = st.slider("彩度", 0.0, 2.0, 1.0)
    dither = st.checkbox("ディザリングを適用", value=True)

    if st.button("変換開始"):
        outline_color_rgb = tuple(int(outline_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        result = process_image(img_array, size, num_colors, contrast, offset, c_min, c_max, outline_color_rgb, saturation, dither)
        
        st.image(result, caption="変換後の画像", use_column_width=True)
        
        result_pil = Image.fromarray(result)
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="画像を保存",
            data=byte_im,
            file_name="high_quality_dot_art.png",
            mime="image/png"
        )

st.write("注意: このアプリは画像処理に時間がかかる場合があります。しばらくお待ちください。")