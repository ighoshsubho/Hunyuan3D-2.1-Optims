from typing import List, Union, Optional
from PIL import Image
from rembg import remove, new_session


class BackgroundRemover:
    def __init__(
        self,
        model_name: str = "bria",
        background_color: List[int] = [255, 255, 255],
        foreground_ratio: float = 0.9,
        auto_crop: bool = True,
        square_output: bool = True,
        use_cuda: bool = True,
        gpu_memory_limit: int = 6 * 1024 * 1024 * 1024,
    ):
        """
        Initialize the BackgroundRemover with configurable options.
        """
        # Ensure all values are proper types
        self.background_color = [int(c) for c in background_color]
        self.foreground_ratio = float(foreground_ratio)
        self.auto_crop = bool(auto_crop)
        self.square_output = bool(square_output)
        
        # Initialize rembg session with optimized settings
        providers = []
        if use_cuda:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": int(gpu_memory_limit),
                    "cudnn_conv_algo_search": "HEURISTIC",
                }
            ))
        providers.append("CPUExecutionProvider")
        
        try:
            self.session = new_session(model_name=model_name, providers=providers)
        except Exception as e:
            print(f"Warning: Could not create session with CUDA, falling back to CPU: {e}")
            self.session = new_session(model_name=model_name, providers=["CPUExecutionProvider"])
    
    def __call__(
        self, 
        images: Union[Image.Image, List[Image.Image]], 
        force_remove: bool = False,
        **rembg_kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Remove background from image(s) with advanced preprocessing.
        """
        is_single_image = isinstance(images, Image.Image)
        if is_single_image:
            images = [images]
        
        processed_images = []
        
        for image in images:
            try:
                processed_image = self._process_single_image(
                    image, force_remove, **rembg_kwargs
                )
                processed_images.append(processed_image)
            except Exception as e:
                print(f"Error processing image: {e}")
                # Return original image if processing fails
                processed_images.append(image)
        
        return processed_images[0] if is_single_image else processed_images
    
    def _process_single_image(
        self, 
        image: Image.Image, 
        force_remove: bool = False,
        **rembg_kwargs
    ) -> Image.Image:
        """Process a single image with background removal and preprocessing."""
        # Ensure image is in correct format
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")
        
        original_size = image.size
        width, height = int(original_size[0]), int(original_size[1])
        
        # Check if we need to remove background
        should_remove_bg = self._should_remove_background(image, force_remove)
        
        if should_remove_bg:
            print("Removing background using AI model...")
            try:
                image = remove(image, session=self.session, **rembg_kwargs)
            except Exception as e:
                print(f"Background removal failed: {e}, using original image")
                return image
        else:
            print("Using existing alpha channel as mask")
        
        # Skip cropping and resizing if not requested
        if not self.auto_crop:
            return self._apply_background_color(image)
        
        # Get alpha channel for cropping
        if image.mode != "RGBA":
            # If no alpha channel, assume the entire image is foreground
            return self._apply_background_color(image)
        
        try:
            alpha = image.split()[-1]
            bbox = alpha.getbbox()
            
            if not bbox:
                # No foreground detected, return original with background
                return self._apply_background_color(image)
            
            # Crop and resize logic
            processed_image = self._crop_and_resize(image, alpha, bbox, (width, height))
            
            # Make square if requested
            if self.square_output:
                processed_image = self._make_square(processed_image)
            
            return processed_image
            
        except Exception as e:
            print(f"Error in cropping/resizing: {e}")
            return self._apply_background_color(image)
    
    def _should_remove_background(self, image: Image.Image, force_remove: bool) -> bool:
        """Determine if background should be removed."""
        if force_remove:
            return True
        
        # Check if image already has meaningful alpha channel
        try:
            if image.mode == "RGBA":
                alpha_extrema = image.getextrema()[3]
                if alpha_extrema[0] < 255:  # Alpha channel has transparency
                    return False
        except Exception:
            pass  # If we can't check, assume we need to remove background
        
        return True
    
    def _crop_and_resize(
        self, 
        image: Image.Image, 
        alpha: Image.Image, 
        bbox: tuple, 
        original_size: tuple
    ) -> Image.Image:
        """Crop image to object bounds and resize according to foreground_ratio."""
        try:
            width, height = int(original_size[0]), int(original_size[1])
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            dy, dx = y2 - y1, x2 - x1
            
            # Ensure we have valid dimensions
            if dy <= 0 or dx <= 0:
                return image
            
            # Calculate scaling factor to fit foreground_ratio
            scale_y = (height * self.foreground_ratio) / dy
            scale_x = (width * self.foreground_ratio) / dx
            scale = min(scale_y, scale_x)
            
            new_height = max(1, int(dy * scale))
            new_width = max(1, int(dx * scale))
            
            # Apply background color
            background = Image.new("RGBA", (width, height), (*self.background_color, 255))
            image = Image.alpha_composite(background, image)
            
            # Crop to object bounds
            cropped_image = image.crop(bbox)
            cropped_alpha = alpha.crop(bbox)
            
            # Resize
            resized_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_alpha = cropped_alpha.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create padded image with original dimensions
            padded_image = Image.new("RGB", (width, height), tuple(self.background_color))
            padded_alpha = Image.new("L", (width, height), 0)
            
            # Calculate paste position (center)
            paste_x = max(0, (width - new_width) // 2)
            paste_y = max(0, (height - new_height) // 2)
            
            padded_image.paste(resized_image, (paste_x, paste_y))
            padded_alpha.paste(resized_alpha, (paste_x, paste_y))
            padded_image.putalpha(padded_alpha)
            
            return padded_image
            
        except Exception as e:
            print(f"Error in crop_and_resize: {e}")
            return image
    
    def _make_square(self, image: Image.Image) -> Image.Image:
        """Convert image to square (1:1) aspect ratio."""
        try:
            width, height = int(image.size[0]), int(image.size[1])
            
            if width == height:
                return image
            
            # Create square canvas
            size = max(width, height)
            square_image = Image.new("RGB", (size, size), tuple(self.background_color))
            square_alpha = Image.new("L", (size, size), 0)
            
            # Calculate paste position (center)
            paste_x = max(0, (size - width) // 2)
            paste_y = max(0, (size - height) // 2)
            
            # Split and paste
            if image.mode == "RGBA":
                rgb_part = image.convert("RGB")
                alpha_part = image.split()[-1]
                square_image.paste(rgb_part, (paste_x, paste_y))
                square_alpha.paste(alpha_part, (paste_x, paste_y))
                square_image.putalpha(square_alpha)
            else:
                square_image.paste(image, (paste_x, paste_y))
            
            return square_image
            
        except Exception as e:
            print(f"Error in make_square: {e}")
            return image
    
    def _apply_background_color(self, image: Image.Image) -> Image.Image:
        """Apply background color to image if it has transparency."""
        try:
            if image.mode != "RGBA":
                return image
            
            background = Image.new("RGB", image.size, tuple(self.background_color))
            return Image.alpha_composite(background.convert("RGBA"), image).convert("RGB")
            
        except Exception as e:
            print(f"Error applying background color: {e}")
            return image.convert("RGB") if image.mode == "RGBA" else image