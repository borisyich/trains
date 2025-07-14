import gradio as gr
import logging
import math
import numpy as np
import ollama
import time
import torch
import warnings
from diffusers import AutoPipelineForInpainting
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

warnings.filterwarnings("ignore")

# Configure logging to output to console and a file for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output logs to console
        logging.FileHandler("app.log")  # Save logs to app.log file
    ]
)
logger = logging.getLogger(__name__)

# Load machine learning models for inpainting and segmentation
def load_models():
    """Initialize and load Stable Diffusion XL Inpainting and SAM models."""
    logger.info("Loading models...")
    start_time = time.time()
    
    # Load Stable Diffusion XL Inpainting model, optimized for GPU if available
    inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inpainting model loaded in {time.time() - start_time:.2f} seconds")
    
    # Load Segment Anything Model (SAM) for automatic segmentation
    sam_checkpoint = "./checkpoints/sam_vit_b_01ec64.pth"
    start_time = time.time()
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    logger.info(f"SAM model loaded in {time.time() - start_time:.2f} seconds")
    
    return inpaint_pipe, sam_predictor

# Calculate image dimensions compatible with model requirements
def get_compatible_size(width, height):
    """Return image dimensions that are multiples of 64, capped at 2048 for VRAM efficiency."""
    max_size = 2048  # Max size to fit within 12GB VRAM
    target_size = min(max(width, height), max_size)
    target_size = math.ceil(target_size / 64) * 64
    return target_size, target_size

# Enhance user prompt using Gemma model via Ollama
def enhance_prompt(user_prompt, temperature, max_tokens):
    """Enhance user-provided prompt for better inpainting results using Gemma model."""
    logger.info(f"Enhancing prompt: {user_prompt}")
    start_time = time.time()
    
    torch.cuda.empty_cache()  # Clear GPU memory to prevent overflow
    instruction = (
        f"""You are an expert in generating detailed prompts for image editing, specializing in aircraft liveries. 
        Take the following user prompt and enhance it by adding specific details like texture, style, and aviation context. 
        Make it concise, realistic, and suitable for Stable Diffusion inpainting.
        Also provide a negative prompt for inpainting.
        User prompt: {user_prompt}"""
    )

    try:
        response = ollama.generate(
            model="gemma3n",
            prompt=instruction,
            options={
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        enhanced_prompt = response["response"].strip()
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        logger.info(f"Prompt enhancement completed in {time.time() - start_time:.2f} seconds")
        return enhanced_prompt
    except Exception as e:
        logger.error(f"Error in Ollama prompt enhancement: {str(e)}")
        raise gr.Error(f"Failed to enhance prompt with Ollama: {str(e)}")

# Perform automatic segmentation using SAM with user-specified click points
def segment_aircraft(image, click_points):
    """Generate a segmentation mask for the aircraft using SAM based on user click points."""
    logger.info(f"Starting segmentation with points: {click_points}")
    start_time = time.time()
    
    if not click_points:
        logger.error("No points provided for SAM")
        raise gr.Error("Please click on the image to select at least one point")
    
    image_np = np.array(image)
    sam_predictor.set_image(image_np)
    
    # Prepare coordinates and labels for SAM
    point_coords = np.array(click_points)
    point_labels = np.ones(len(click_points))  # All points are positive
    
    masks, _, _ = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )
    
    # Combine masks using logical OR to create a single mask
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    mask_image = Image.fromarray(combined_mask)
    
    logger.info(f"Segmentation completed in {time.time() - start_time:.2f} seconds")
    return mask_image

# Handle user clicks on the image to capture coordinates
def get_click_coordinates(image, evt: gr.SelectData, click_points):
    """Capture and store coordinates where the user clicks on the image."""
    if image is None:
        logger.error("No image provided")
        return None, [], "Please upload an image"
    
    x, y = evt.index[0], evt.index[1]
    orig_width, orig_height = image.size
    
    # Append new click point to the list
    click_points = click_points or []
    click_points.append([x, y])
    
    logger.info(f"Image clicked at coordinates ({x}, {y}) on size ({orig_width}, {orig_height})")
    logger.info(f"Current click points: {click_points}")
    
    return (orig_width, orig_height), click_points, f"Selected points: {click_points}"

# Clear the list of stored click points
def clear_click_points():
    """Reset the list of click points for SAM segmentation."""
    logger.info("Clearing click points")
    return [], "No points selected"

# Generate a preview of the segmentation mask
def preview_mask(image, click_points, orig_size):
    """Generate and display a preview of the SAM-generated mask."""
    logger.info(f"Previewing mask with points: {click_points}")
    if image is None:
        logger.error("No image provided")
        raise gr.Error("Please upload an image")
    if not click_points:
        logger.error("No points provided for SAM")
        raise gr.Error("Please click on the image to select at least one point")
    
    orig_width, orig_height = image.size if orig_size is None else orig_size
    proc_width, proc_height = get_compatible_size(orig_width, orig_height)
    proc_image = image.resize((proc_width, proc_height))
    
    scale_x = proc_width / orig_width
    scale_y = proc_height / orig_height
    adjusted_points = [[x * scale_x, y * scale_y] for x, y in click_points]
    logger.info(f"Adjusted coordinates for SAM preview: {adjusted_points}")
    
    mask = segment_aircraft(proc_image, adjusted_points)
    mask = mask.resize((orig_width, orig_height))
    return mask

# Extract mask from composite image (manual sketch mode)
def extract_mask_from_composite(background, composite):
    """Extract a mask by comparing the composite image with the background."""
    logger.info("Extracting mask from composite and background")
    start_time = time.time()
    
    # Convert numpy arrays to PIL images if necessary
    if isinstance(background, np.ndarray):
        background = Image.fromarray(background).convert("RGB")
    if isinstance(composite, np.ndarray):
        composite = Image.fromarray(composite).convert("RGB")
    
    # Convert to numpy arrays for difference calculation
    bg_array = np.array(background.convert("RGB"))
    comp_array = np.array(composite.convert("RGB"))
    
    # Check for shape mismatch
    if bg_array.shape != comp_array.shape:
        logger.error(f"Shape mismatch: background {bg_array.shape}, composite {comp_array.shape}")
        raise gr.Error("Background and composite images have different sizes.")
    
    # Calculate difference between composite and background
    diff = np.abs(comp_array - bg_array)
    diff_sum = np.sum(diff, axis=2)
    mask_array = (diff_sum > 0).astype(np.uint8) * 255
    
    # Validate mask contains non-zero pixels
    if mask_array.max() == 0:
        logger.error("Extracted mask is empty (no differences between composite and background)")
        raise gr.Error("The drawn mask is empty. Please draw a white or colored area on the image.")
    
    mask = Image.fromarray(mask_array).convert("L")
    logger.info(f"Mask extraction completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Extracted mask shape: {mask_array.shape}, max value: {mask_array.max()}")
    return mask

# Perform inpainting to edit the aircraft image
def edit_aircraft(
        image, 
        click_points, 
        orig_size, 
        manual_mask, mask_mode, 
        user_prompt, negative_prompt, 
        enhanced_prompt, 
        temperature, max_tokens, num_inference_steps, guidance_scale, blur_factor
    ):
    """Edit the aircraft image by applying inpainting based on user inputs and mask."""
    logger.info(f"Starting image editing with user prompt: {user_prompt}, enhanced prompt: {enhanced_prompt}, mask mode: {mask_mode}")
    logger.info(f"Click points: {click_points}, Manual mask from ImageEditor: {manual_mask}")
    logger.info(f"Hyperparameters: temperature={temperature}, max_tokens={max_tokens}, num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale}, blur_factor={blur_factor}")
    start_time = time.time()
    
    if image is None:
        logger.error("No image provided")
        raise ValueError("Please upload an image")
    
    orig_width, orig_height = image.size if orig_size is None else orig_size
    logger.info(f"Original image size: {orig_width}x{orig_height}")
    
    # Use enhanced prompt if available, otherwise fall back to user prompt
    final_prompt = enhanced_prompt if enhanced_prompt else user_prompt
    if not final_prompt:
        logger.warning("No prompt provided, using default")
        final_prompt = "realistic aircraft livery, high detail"
    logger.info(f"Using prompt for inpainting: {final_prompt}")
    
    proc_width, proc_height = get_compatible_size(orig_width, orig_height)
    logger.info(f"Processing image at size: {proc_width}x{proc_height}")
    
    proc_image = image.resize((proc_width, proc_height))
    
    mask_start = time.time()
    if mask_mode == "Point-based (SAM)":
        if not click_points:
            logger.error("No points selected for SAM")
            raise gr.Error("Please click on the image to select at least one point")
        
        scale_x = proc_width / orig_width
        scale_y = proc_height / orig_height
        adjusted_points = [[x * scale_x, y * scale_y] for x, y in click_points]
        logger.info(f"Adjusted coordinates for SAM: {adjusted_points}")
        
        mask = segment_aircraft(proc_image, adjusted_points)
        display_mask = mask
    else:  # Manual (Sketch)
        mask = extract_mask_from_composite(manual_mask["background"], manual_mask["composite"])
        display_mask = mask
        mask = mask.resize((proc_width, proc_height))
        logger.info("Using manually drawn mask")
    
    # Apply blur to mask for smoother transitions
    mask = inpaint_pipe.mask_processor.blur(mask, blur_factor=blur_factor)
    logger.info(f"Applied blur to mask with blur_factor={blur_factor}")

    logger.info(f"Mask generated in {time.time() - mask_start:.2f} seconds")
    
    logger.info("Starting inpainting...")
    inpaint_start = time.time()
    torch.cuda.empty_cache()  # Clear GPU memory

    def progress_callback(step, timestep, latents):
        logger.info(f"Inpainting step {step}/{num_inference_steps}")

    edited_image = inpaint_pipe(
        prompt=final_prompt,
        image=proc_image,
        mask_image=mask,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps,
        negative_prompt=negative_prompt if negative_prompt else "blurry, low quality, distorted, unrealistic, cartoonish, low detail",
        callback=progress_callback,
        callback_steps=10
    ).images[0]
    logger.info(f"Inpainting completed in {time.time() - inpaint_start:.2f} seconds")
    
    # Resize output to original dimensions
    edited_image = edited_image.resize((orig_width, orig_height))
    display_mask = display_mask.resize((orig_width, orig_height))
    logger.info(f"Resized output and mask to original size: {orig_width}x{orig_height}")

    logger.info(f"Total editing time: {time.time() - start_time:.2f} seconds")
    return edited_image, display_mask, enhanced_prompt

# Initialize models
inpaint_pipe, sam_predictor = load_models()

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Aircraft Paint Editor")
    gr.Markdown("Upload an aircraft image, choose a mask mode (point-based or manual), click or draw on the image, and describe the new paint or livery.")
    gr.Markdown("Example prompts: 'Glossy black fuselage with gold horizontal stripes, realistic aviation paint' or 'Military green camouflage with gray accents, matte finish'")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Aircraft Image", type="pil", interactive=True)
            mask_mode = gr.Radio(
                choices=["Point-based (SAM)", "Manual (Sketch)"],
                label="Mask Selection Mode",
                value="Manual (Sketch)"
            )
            click_coords = gr.Textbox(label="Selected Point Coordinates (for SAM)", value="No points selected")
            clear_points_button = gr.Button("Clear Points")
            preview_mask_button = gr.Button("Preview Mask", visible=False)
            manual_mask_input = gr.ImageEditor(label="Draw Mask", interactive=True, visible=True)
            user_prompt = gr.Textbox(label="Describe new paint/livery (e.g., 'blue fuselage with red stripes')")
            negative_prompt = gr.Textbox(label="Negative Prompt (e.g., 'blurry, low quality, unrealistic')")
            enhanced_prompt = gr.Textbox(label="Enhanced Prompt (editable)", interactive=True)
            with gr.Group():
                gr.Markdown("### Prompt Enhancement Parameters")
                temperature = gr.Slider(label="Temperature (Gemma creativity)", minimum=0.1, maximum=1.5, step=0.1, value=0.7)
                max_tokens = gr.Slider(label="Max Tokens (Gemma prompt length)", minimum=50, maximum=500, step=10, value=100)
            with gr.Group():
                gr.Markdown("### Inpainting Parameters")
                num_inference_steps = gr.Slider(label="Inference Steps (Diffusion quality)", minimum=20, maximum=100, step=1, value=30)
                guidance_scale = gr.Slider(label="Guidance Scale (Prompt adherence)", minimum=5, maximum=20, step=1, value=10)
                blur_factor = gr.Slider(label="Mask Blur Factor (Edge smoothness)", minimum=0, maximum=100, step=1, value=50)
            enhance_button = gr.Button("Enhance Prompt")
            edit_button = gr.Button("Edit")
        with gr.Column():
            mask_output = gr.Image(label="Generated Mask")
            edited_output = gr.Image(label="Edited Aircraft")
    
    # States for storing click points and original image size
    click_points_state = gr.State(value=[])
    orig_size_state = gr.State()
    
    # Update visibility of mask input and preview button based on mask mode
    def update_mask_input_visibility(mode):
        is_manual = mode == "Manual (Sketch)"
        return gr.update(visible=is_manual), gr.update(visible=not is_manual)
    
    mask_mode.change(
        fn=update_mask_input_visibility,
        inputs=mask_mode,
        outputs=[manual_mask_input, preview_mask_button]
    )
    
    # Handle image click events
    image_input.select(
        fn=get_click_coordinates,
        inputs=[image_input, click_points_state],
        outputs=[orig_size_state, click_points_state, click_coords]
    )
    
    # Clear click points
    clear_points_button.click(
        fn=clear_click_points,
        inputs=None,
        outputs=[click_points_state, click_coords]
    )
    
    # Preview segmentation mask
    preview_mask_button.click(
        fn=preview_mask,
        inputs=[image_input, click_points_state, orig_size_state],
        outputs=mask_output
    )

    # Enhance user prompt
    enhance_button.click(
        fn=enhance_prompt,
        inputs=user_prompt,
        outputs=enhanced_prompt
    )
    
    # Perform image editing
    edit_button.click(
        fn=edit_aircraft,
        inputs=[
            image_input, click_points_state, orig_size_state, 
            manual_mask_input, mask_mode, 
            user_prompt, negative_prompt, enhanced_prompt,
            temperature, max_tokens,
            num_inference_steps, guidance_scale, blur_factor
        ],
        outputs=[edited_output, mask_output, enhanced_prompt]
    )

# Launch the Gradio interface
logger.info("Launching Gradio interface...")
demo.launch()