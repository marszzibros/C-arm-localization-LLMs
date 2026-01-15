import os
import glob
from PIL import Image, ImageDraw, ImageFont

def create_gifs():
    identifiers = ['1_4', '1_5', '4_1', '5_1']
    models = ['base', 'finetune']
    paths = ['path1', 'path2', 'path3']

    # Ensure we are in the correct directory or using absolute paths
    # Assuming script is run from results/navigation/

    for model in models:
        for ident in identifiers:
            images = []
            print(f"Processing {model} - {ident}...")
            
            # 1. Find Init Image
            # Pattern: init/{ident}*.png
            init_search = os.path.join("init", f"{ident}*.png")
            init_files = glob.glob(init_search)
            
            if init_files:
                # Take the first match (assuming unique prefix enough to identify or we just take first)
                init_path = init_files[0]
                try:
                    img = Image.open(init_path)
                    images.append(img)
                    print(f"  Found init: {init_path}")
                except Exception as e:
                    print(f"  Error opening {init_path}: {e}")
            else:
                print(f"  Warning: No init image found for {ident}")

            # 2. Find Path Images
            for path in paths:
                # Pattern: {model}/{path}/{ident}*.png
                path_search = os.path.join(model, path, f"{ident}*.png")
                path_files = glob.glob(path_search)
                
                if path_files:
                    p_path = path_files[0]
                    try:
                        img = Image.open(p_path)
                        images.append(img)
                        print(f"  Found path step: {p_path}")
                    except Exception as e:
                        print(f"  Error opening {p_path}: {e}")
                else:
                    print(f"  Warning: No image found in {model}/{path} for {ident}")

            # 3. Save GIF
            if len(images) > 0:
                output_filename = f"{model}_{ident}.gif"
                try:
                    # Save as GIF
                    # duration is in milliseconds. 500ms = 0.5s
                    
                    # Add order number to images
                    processed_images = []
                    for idx, img in enumerate(images):
                        try:
                            # Convert to RGB (in case of RGBA/P palette) to ensure drawing works reliably
                            img_copy = img.convert("RGB")
                            draw = ImageDraw.Draw(img_copy)
                            
                            # Calculate font size based on image size (e.g., 20% of height)
                            # Default to a reasonable size if dynamic sizing fails
                            width, height = img_copy.size
                            font_size = int(height * 0.2) 
                            
                            # Try to load a font, fallback to default if not available
                            try:
                                # Try standard fonts (Linux)
                                font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                            except IOError:
                                try:
                                    font = ImageFont.truetype("arial.ttf", font_size)
                                except IOError:
                                    # Fallback to default (might be small)
                                    font = ImageFont.load_default()
                                    # Since default font is tiny and can't be scaled, we might rely on the large font loading
                                    print("  Warning: Could not load TrueType font, using default (small).")
                            
                            text = str(idx + 1)
                            
                            # Get text position (Top Right)
                            # Using textbbox if available (Pillow >= 8.0.0), else basic estimation
                            try:
                                bbox = draw.textbbox((0, 0), text, font=font)
                                text_w = bbox[2] - bbox[0]
                                text_h = bbox[3] - bbox[1]
                            except AttributeError:
                                # Older Pillow
                                text_w, text_h = draw.textsize(text, font=font)
                            
                            x = width - text_w - 20 # 20px padding from right
                            y = 10 # 10px padding from top
                            
                            # Draw Red Text
                            draw.text((x, y), text, fill="red", font=font)
                            
                            processed_images.append(img_copy)
                        except Exception as e:
                            print(f"  Error drawing text on frame {idx}: {e}")
                            processed_images.append(img) # Fallback to original

                    processed_images[0].save(
                        output_filename,
                        save_all=True,
                        append_images=processed_images[1:],
                        duration=500,
                        loop=0
                    )
                    print(f"  Successfully created {output_filename}")
                except Exception as e:
                    print(f"  Error saving GIF {output_filename}: {e}")
            else:
                print(f"  Skipping {model}_{ident} (no images found)")
            print("-" * 20)

if __name__ == "__main__":
    create_gifs()
