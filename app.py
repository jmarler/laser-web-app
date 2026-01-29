import gc
from io import BytesIO
import base64
from typing import Tuple, List

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, ImageChops, ImageFilter
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from rembg import remove

# ==========================================
# 1. HARDWARE ACCELERATION SETUP
# ==========================================
def get_device_config():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        # MPS (Mac) requires Float32 to avoid black images/NaNs
        return "mps", torch.float32
    else:
        return "cpu", torch.float32

DEVICE, DTYPE = get_device_config()

# Global Model Cache
if "txt2img_pipe" not in st.session_state:
    st.session_state["txt2img_pipe"] = None

def cleanup_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()

# ==========================================
# 2. IMAGE & VECTOR UTILITIES
# ==========================================
def pil_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def mask_to_svg(mask: Image.Image, mode: str = "stroke") -> str:
    """
    Converts a binary PIL mask into an SVG string with simplification for cleaner vectors.
    """
    mask_np = np.array(mask.convert("L"))
    _, thresh = cv2.threshold(mask_np, 127, 255, 0)
    
    # RETR_TREE captures holes
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = mask_np.shape
    svg_parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    
    for contour in contours:
        # Vector Smoothing (Douglas-Peucker algorithm)
        # 0.1% to 0.2% of arc length is usually a good balance.
        epsilon = 0.002 * cv2.arcLength(contour, True)
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        path_data = []
        # Check if contour collapsed
        if len(smoothed_contour) < 3:
            continue

        for i, point in enumerate(smoothed_contour):
            x, y = point[0]
            if i == 0:
                path_data.append(f"M {x} {y}")
            else:
                path_data.append(f"L {x} {y}")
        path_data.append("Z")
        
        d = " ".join(path_data)
        
        if mode == "stroke":
            svg_parts.append(f'<path d="{d}" stroke="red" stroke-width="1" fill="none" />')
        else:
            svg_parts.append(f'<path d="{d}" stroke="none" fill="black" />')
            
    svg_parts.append('</svg>')
    return "".join(svg_parts)

def get_outline_image(mask: Image.Image) -> Image.Image:
    mask_np = np.array(mask.convert("L"))
    _, thresh = cv2.threshold(mask_np, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    outline_img = np.zeros_like(mask_np)
    cv2.drawContours(outline_img, contours, -1, (255), 2)
    return Image.fromarray(outline_img)

# ==========================================
# 3. CORE LOGIC
# ==========================================

def remove_background(
    input_image: Image.Image, 
    mode: str = "ai", 
    threshold_val: int = 127,
    smoothing: float = 0.0,
    expansion: int = 0,
    invert_detection: bool = False
) -> Tuple[Image.Image, Image.Image]:
    """
    Extracts the subject with advanced controls:
    - threshold_val: Manual brightness cutoff
    - smoothing: Blur amount before re-thresholding
    - expansion: Morphological Dilation (+ values) or Erosion (- values)
    - invert_detection: Flips what is considered foreground
    """
    if mode == "silhouette":
        # 1. Convert to Greyscale
        gray = input_image.convert("L")
        
        # 2. Manual Thresholding
        # Standard: Pixels > Threshold = White (255), else Black (0)
        mask_rough = gray.point(lambda p: 255 if p > threshold_val else 0)
        
        if invert_detection:
            mask_rough = ImageOps.invert(mask_rough)

    else:
        # Standard AI Mode (rembg)
        result = remove(input_image)
        mask_rough = result.split()[3]
        
        # rembg usually returns strict alpha 0 or 255, but sometimes gradients.
        # Ensure binary.
        mask_rough = mask_rough.point(lambda p: 255 if p > 10 else 0)

    # Convert to Numpy for Morphological Ops (Thickness)
    mask_np = np.array(mask_rough)

    # 3. Expansion / Erosion (Thickness)
    if expansion != 0:
        kernel_size = abs(expansion) * 2 + 1 # odd number
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if expansion > 0:
            # Dilate (Thicken / Fill Holes)
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        else:
            # Erode (Thin / Trim Edges)
            mask_np = cv2.erode(mask_np, kernel, iterations=1)

    # Convert back to PIL for Smoothing
    mask_processed = Image.fromarray(mask_np)

    # 4. Smoothing (Blur & Snap)
    if smoothing > 0:
        # Blur melts the jagged edges together
        mask_blurred = mask_processed.filter(ImageFilter.GaussianBlur(radius=smoothing))
        # Re-threshold snaps it back to a clean vector-like line
        # 127 is the midpoint; adjusting this can favor expansion/contraction slightly
        mask_final = mask_blurred.point(lambda p: 255 if p > 127 else 0)
    else:
        mask_final = mask_processed

    # 5. Crop to Content
    bbox = mask_final.getbbox()
    if bbox:
        cropped_image = input_image.crop(bbox)
        cropped_mask = mask_final.crop(bbox)
        return cropped_mask, cropped_image
    
    return mask_final, input_image

def generate_pattern(prompt: str, seed: int) -> Image.Image:
    if st.session_state["txt2img_pipe"] is None:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=DTYPE,
            safety_checker=None, 
            requires_safety_checker=False
        ).to(DEVICE)
        
        if DEVICE == "mps":
            pipe.vae.to(dtype=torch.float32)

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        if DEVICE == "mps":
            pipe.enable_attention_slicing()
        
        st.session_state["txt2img_pipe"] = pipe
    
    pipe = st.session_state["txt2img_pipe"]
    
    engineered_prompt = (
        f"{prompt}, greyscale depth map, 3d relief style, height map, "
        "black and white, high contrast, sharp edges, vector style, no background, "
        "intricate details, centered, symmetrical"
    )
    negative_prompt = "color, blur, low quality, fuzz, dithering, gradient, realistic photo, shadows, watermark"
    
    generator = torch.Generator(DEVICE).manual_seed(seed)
    
    image = pipe(
        prompt=engineered_prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_inference_steps=25,
        generator=generator
    ).images[0]
    
    cleanup_memory()
    return image

def apply_pattern_transform(
    shape_mask: Image.Image, 
    pattern: Image.Image, 
    zoom: float = 1.0, 
    pan_x: int = 0, 
    pan_y: int = 0
) -> Image.Image:
    w_shape, h_shape = shape_mask.size
    w_pat, h_pat = pattern.size

    # Base Scale
    target_ratio = w_shape / h_shape
    pattern_ratio = w_pat / h_pat
    
    if target_ratio > pattern_ratio:
        base_w = w_shape
        base_h = int(w_shape / pattern_ratio)
    else:
        base_w = int(h_shape * pattern_ratio)
        base_h = h_shape
        
    if base_w < w_shape:
        s = w_shape / base_w
        base_w, base_h = int(base_w * s), int(base_h * s)
    if base_h < h_shape:
        s = h_shape / base_h
        base_w, base_h = int(base_w * s), int(base_h * s)

    # Zoom
    final_w = int(base_w * zoom)
    final_h = int(base_h * zoom)
    
    resized_pattern = pattern.resize((final_w, final_h), Image.LANCZOS)
    
    # Tiling
    canvas_w = max(w_shape, final_w) * 2
    canvas_h = max(h_shape, final_h) * 2
    
    tiled_canvas = Image.new("L", (canvas_w, canvas_h))
    
    for y in range(0, canvas_h, final_h):
        for x in range(0, canvas_w, final_w):
            tiled_canvas.paste(resized_pattern.convert("L"), (x, y))

    # Crop with Pan
    center_x = canvas_w // 2
    center_y = canvas_h // 2
    
    crop_x = (center_x - w_shape // 2) - pan_x
    crop_y = (center_y - h_shape // 2) - pan_y
    
    pattern_layer = tiled_canvas.crop((crop_x, crop_y, crop_x + w_shape, crop_y + h_shape))

    # Masking
    final_img = Image.new("L", (w_shape, h_shape), 0)
    final_img.paste(pattern_layer, (0, 0), shape_mask)
    
    return final_img

# ==========================================
# 4. USER INTERFACE
# ==========================================
st.set_page_config(layout="wide", page_title="Laser Depth Generator")

st.title("Laser Engraving Workflow")
st.markdown(f"**Device:** `{DEVICE.upper()}` | **Precision:** `{DTYPE}`")

with st.sidebar:
    st.header("1. Input Mode")
    input_mode = st.radio(
        "Source Type", 
        ["Standard Photo (AI Remove)", "Silhouette / Mask"],
        help="Use 'Silhouette' for manual control over Thresholding."
    )
    
    mode_key = "ai" if "Standard" in input_mode else "silhouette"
    
    # --- Advanced Masking Controls ---
    st.subheader("Mask Tuning")
    
    # Default values
    threshold_val = 127
    smoothing_val = 0.0
    expansion_val = 0
    invert_det = False
    
    if mode_key == "silhouette":
        threshold_val = st.slider("Brightness Threshold", 0, 255, 127, 
            help="Adjust to separate object from background. Lower = More sensitive.")
        
        invert_det = st.checkbox("Invert Detection", value=False, 
            help="Check this if the background is being detected instead of the object.")
    
    smoothing_val = st.slider("Edge Smoothing", 0.0, 25.0, 2.0, step=0.5, 
        help="Higher values melt jagged edges into smooth curves.")
        
    expansion_val = st.slider("Mask Expansion (Thickness)", -10, 10, 0, 
        help="Positive = Thicker (fills holes). Negative = Thinner (trims edges).")
    
    st.divider()

    st.header("2. Pattern Settings")
    prompt = st.text_input("Describe Pattern", "intricate celtic knotwork pattern")
    seed = st.number_input("Random Seed", value=1234, step=1)
    
    st.header("3. Laser Settings")
    invert = st.checkbox("Invert (White=Deep)", value=False)
    brightness = st.slider("Brightness", -50, 50, 0)
    contrast = st.slider("Contrast", 1.0, 3.0, 1.0)
    
    st.divider()
    
    st.header("4. Pattern Transform")
    st.info("Adjust these after generating.")
    zoom = st.slider("Zoom Pattern", 0.1, 3.0, 1.0, 0.1)
    pan_x = st.number_input("Pan X", value=0, step=10)
    pan_y = st.number_input("Pan Y", value=0, step=10)

uploaded_file = st.file_uploader("Upload Object Photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("Extracting shape..."):
        # Process Mask
        mask, cropped_image = remove_background(
            raw_image, 
            mode=mode_key, 
            threshold_val=threshold_val,
            smoothing=smoothing_val,
            expansion=expansion_val,
            invert_detection=invert_det
        )
        outline_vis = get_outline_image(mask)
        
        svg_cut = mask_to_svg(mask, mode="stroke")
        svg_fill = mask_to_svg(mask, mode="fill")

    # --- UI ROW 1 ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.image(cropped_image, caption="1. Input (Cropped)")
        
    with c2:
        st.image(mask, caption=f"2. Mask (Thresh: {threshold_val}, Smooth: {smoothing_val})")
        st.download_button("Download Mask SVG", svg_fill, "mask_fill.svg", "image/svg+xml")

    with c3:
        st.image(outline_vis, caption="3. Vector Outline Check")
        st.download_button("Download Cut SVG", svg_cut, "cut_line.svg", "image/svg+xml")
    
    st.divider()

    # --- Generation ---
    if "raw_texture" not in st.session_state:
        st.session_state["raw_texture"] = None

    if st.button("Generate Laser Pattern", type="primary"):
        with st.status("Generating...", expanded=True):
            st.write("🎨 Generating high-contrast height map...")
            generated = generate_pattern(prompt, seed)
            
            if not generated.getbbox():
                st.error("Error: AI generated black image.")
                st.stop()
                
            st.session_state["raw_texture"] = generated
            st.write("Done!")
    
    if st.session_state["raw_texture"] is not None:
        raw_texture = st.session_state["raw_texture"]
        
        final_map = apply_pattern_transform(mask, raw_texture, zoom, pan_x, pan_y)
        
        if contrast != 1.0 or brightness != 0:
            enhancer = ImageOps.autocontrast(final_map, cutoff=0)
            final_map = Image.fromarray(np.clip(np.array(final_map) * contrast + brightness, 0, 255).astype(np.uint8))
        
        if invert:
            final_map = ImageOps.invert(final_map)

        r1, r2 = st.columns(2)
        with r1:
            st.image(raw_texture, caption="Base Texture")
        with r2:
            st.image(final_map, caption="Final Engraving File")
            
        st.download_button(
            label="Download Laser Ready PNG",
            data=pil_to_bytes(final_map),
            file_name="laser_engraving.png",
            mime="image/png"
        )