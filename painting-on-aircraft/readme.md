# Painting-on-aircraft

## Overview
The **Aircraft Paint Editor** is a Python application that leverages advanced machine learning models to edit aircraft liver Livery through inpainting. It uses **Stable Diffusion XL Inpainting** for image editing and the **Segment Anything Model (SAM)** for automatic segmentation of aircraft regions. The application provides a user-friendly **Gradio** interface to upload images, select regions (via points or manual sketching), and describe desired paint schemes, which are enhanced by the **Gemma** model through **Ollama**.

## Features
- **Image Upload**: Upload an aircraft image for editing.
- **Mask Selection Modes**:
  - **Point-based (SAM)**: Click points on the image to automatically segment the aircraft using SAM.
  - **Manual (Sketch)**: Draw directly on the image to define the region for inpainting.
- **Prompt Enhancement**: Improve user prompts for inpainting using Gemma via Ollama, tailored for realistic aircraft liveries.
- **Inpainting Parameters**: Customize diffusion quality, prompt adherence, and mask edge smoothness.
- **Logging**: Detailed logs for debugging and performance tracking, saved to `app.log`.

## Requirements
- Python 3.8+
- Libraries:
  - `gradio`: For the web interface.
  - `torch`: For GPU-accelerated model inference.
  - `diffusers`: For Stable Diffusion XL Inpainting.
  - `segment_anything`: For SAM-based segmentation.
  - `ollama`: For prompt enhancement with Gemma.
  - `numpy`, `PIL`: For image processing.
- Hardware:
  - GPU with at least 12GB VRAM recommended for optimal performance.
  - SAM checkpoint file (`sam_vit_b_01ec64.pth`) in the `checkpoints/` directory.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/borisyich/trains.git
   cd painting-on-aircraft
   ```
2. Install dependencies:

3. Download the SAM checkpoint (`sam_vit_b_01ec64.pth`) and place it in the `checkpoints/` directory.
4. Ensure Ollama is installed and the `gemma3n` model is available:
   ```bash
   ollama pull gemma3n
   ```

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Open the Gradio interface in your browser (URL provided in the console).
3. **Steps**:
   - Upload an aircraft image.
   - Choose a mask mode:
     - **Point-based (SAM)**: Click on the aircraft to select points, then preview the mask.
     - **Manual (Sketch)**: Draw the desired region on the image.
   - Enter a prompt describing the new livery (e.g., "Glossy black fuselage with gold stripes").
   - Optionally enhance the prompt using the "Enhance Prompt" button.
   - Adjust inpainting parameters (e.g., inference steps, guidance scale, blur factor).
   - Click "Edit" to generate the edited image and mask.
4. View the results in the output panel.

## Example Prompts
- "Glossy black fuselage with gold horizontal stripes, realistic aviation paint"
- "Military green camouflage with gray accents, matte finish"

## Project Structure
- `app.py`: Main application script containing the Gradio interface and logic.
- `checkpoints/`: Directory for SAM model checkpoint (`sam_vit_b_01ec64.pth`).
- `app.log`: Log file for debugging and performance tracking.

## Notes
- The application optimizes image sizes to be multiples of 64 and caps at 2048 pixels to manage VRAM usage.
- Ensure sufficient GPU memory when using CUDA; otherwise, it falls back to CPU.
- The SAM model requires click points for point-based segmentation; manual mode requires a drawn mask.
- Negative prompts help improve inpainting quality by excluding undesired features (e.g., "blurry, low quality").
