# Eric Laser Webapp v2

A web application for laser engraving workflow automation. Upload a photo, generate AI-powered texture patterns, and export laser-ready files.

## Overview

This application streamlines the laser engraving workflow by combining:

- **AI Background Removal** - Automatically extract subjects from photos using the rembg library
- **Manual Silhouette Mode** - Fine-tune mask extraction with threshold controls for tricky images
- **AI Pattern Generation** - Create texture patterns from text descriptions using Stable Diffusion v1.5
- **Laser-Ready Export** - Download PNG depth maps and SVG cut lines ready for your laser engraver

## How It Works

### Architecture

The app is built with:
- **Streamlit** - Interactive web UI
- **PyTorch** - Deep learning backend with automatic hardware detection (CUDA, MPS, CPU)
- **Stable Diffusion v1.5** - Text-to-image pattern generation optimized for greyscale depth maps
- **rembg** - Neural network-based background removal
- **OpenCV** - Image processing and SVG vector generation

### Processing Pipeline

1. **Image Upload** - Accept JPG/PNG photos
2. **Background Removal** - Extract subject using AI or manual threshold
3. **Mask Refinement** - Apply smoothing, expansion/erosion, and edge cleanup
4. **Pattern Generation** - Generate texture from text prompt using Stable Diffusion
5. **Pattern Transform** - Zoom, pan, and tile the pattern to fit your shape
6. **Laser Adjustments** - Tune brightness, contrast, and inversion for your machine
7. **Export** - Download PNG depth map and SVG cut lines

## How to Use

1. **Upload an image** containing the object you want to engrave

2. **Choose input mode:**
   - **Standard Photo (AI Remove)** - Best for photos with clear subjects
   - **Silhouette / Mask** - Use for high-contrast images or manual control

3. **Tune the mask** (if needed):
   - Brightness Threshold - Adjust separation point (silhouette mode only)
   - Edge Smoothing - Soften jagged edges
   - Mask Expansion - Thicken (+) or thin (-) the mask

4. **Describe your pattern** - Enter a text prompt like "intricate celtic knotwork" or "wood grain texture"

5. **Generate** - Click "Generate Laser Pattern" to create the texture

6. **Adjust the pattern:**
   - Zoom - Scale the pattern up or down
   - Pan X/Y - Reposition the pattern within the shape

7. **Configure laser settings:**
   - Brightness/Contrast - Fine-tune the depth map
   - Invert - Swap black/white if your laser expects the opposite

8. **Download outputs:**
   - **Laser Ready PNG** - Greyscale depth map for engraving
   - **Cut SVG** - Vector outline for cutting
   - **Mask SVG** - Filled shape vector

## Building and Running

### Prerequisites

- Docker and Docker Compose installed
- For GPU acceleration: NVIDIA GPU with drivers and nvidia-container-toolkit (Linux)

### Production (Linux with NVIDIA GPU)

```bash
# Build and start
docker compose up --build

# Or run in background
docker compose up -d --build
```

This configuration:
- Enables full NVIDIA GPU acceleration
- Uses a persistent volume for Hugging Face model cache
- Allocates 2GB shared memory for PyTorch

### Development (Linux with NVIDIA GPU)

```bash
# Build and start with live code reloading
docker compose -f docker-compose.dev.yml up --build
```

This configuration:
- Mounts the local directory for live code changes
- Enables full NVIDIA GPU acceleration
- Uses a persistent volume for model cache

### macOS

```bash
# Build and start
docker compose -f docker-compose.mac.yml up --build
```

This configuration:
- Runs without GPU (uses CPU or MPS if available on host)
- Mounts the local directory for development
- Uses a persistent volume for model cache

### Accessing the App

Once running, open your browser to:

```
http://localhost:8501
```

### Stopping the App

```bash
# If running in foreground, press Ctrl+C

# If running in background
docker compose down

# Or for dev/mac configs
docker compose -f docker-compose.dev.yml down
docker compose -f docker-compose.mac.yml down
```

## System Requirements

### Minimum
- 8GB RAM
- 10GB disk space (for Docker image and model cache)

### Recommended
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM (for fast pattern generation)
- SSD storage

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| ML Framework | PyTorch |
| Image Generation | Stable Diffusion v1.5 (Hugging Face Diffusers) |
| Background Removal | rembg |
| Image Processing | OpenCV, Pillow |
| Containerization | Docker, NVIDIA CUDA 11.8 |
