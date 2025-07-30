"""
Screen Capture Service for ADAM - Enables "seeing" the user's screen
"""
import io
import os
import time
import threading
from typing import Optional, Callable, Tuple, Any
from datetime import datetime
from pathlib import Path
import base64
import logging

# Try to import screen capture libraries
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("Warning: mss not installed. Run: pip install mss")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Run: pip install Pillow")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. Run: pip install pytesseract")

# For macOS, we might need pyobjc
try:
    import AppKit
    import Quartz
    MACOS_AVAILABLE = True
except ImportError:
    MACOS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScreenCaptureService:
    """
    Service for capturing screen content to enable ADAM to "see" what the user sees.
    Supports full screen, active window, and region capture.
    """
    
    def __init__(self, storage_path: str = "./data/screen_captures"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.monitoring = False
        self.monitor_thread = None
        
        # Check available capture methods
        self.capture_method = self._determine_capture_method()
        logger.info(f"Screen capture initialized with method: {self.capture_method}")
    
    def _determine_capture_method(self) -> str:
        """Determine the best capture method for the platform"""
        if MSS_AVAILABLE:
            return "mss"
        elif MACOS_AVAILABLE:
            return "macos"
        else:
            return "none"
    
    def capture_screen(self, monitor_number: int = 1) -> Optional[bytes]:
        """
        Capture the entire screen.
        Returns image data as bytes suitable for vision model input.
        """
        if self.capture_method == "mss" and MSS_AVAILABLE:
            with mss.mss() as sct:
                monitor = sct.monitors[monitor_number]
                screenshot = sct.grab(monitor)
                
                if PIL_AVAILABLE:
                    # Convert to PIL Image
                    img = Image.frombytes('RGB', 
                                        (screenshot.width, screenshot.height), 
                                        screenshot.rgb)
                    
                    # Convert to bytes
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    return img_byte_arr.getvalue()
                else:
                    # Return raw bytes
                    return screenshot.rgb
        
        elif self.capture_method == "macos" and MACOS_AVAILABLE:
            return self._capture_screen_macos()
        
        logger.warning("No screen capture method available")
        return None
    
    def capture_active_window(self) -> Optional[bytes]:
        """
        Capture only the currently active window.
        More focused than full screen capture.
        """
        # This is platform-specific and complex to implement
        # For now, fallback to screen capture
        # TODO: Implement proper active window detection
        return self.capture_screen()
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[bytes]:
        """
        Capture a specific region of the screen.
        Useful for focusing on particular UI elements.
        """
        if self.capture_method == "mss" and MSS_AVAILABLE:
            with mss.mss() as sct:
                monitor = {"top": y, "left": x, "width": width, "height": height}
                screenshot = sct.grab(monitor)
                
                if PIL_AVAILABLE:
                    img = Image.frombytes('RGB', 
                                        (screenshot.width, screenshot.height), 
                                        screenshot.rgb)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    return img_byte_arr.getvalue()
                else:
                    return screenshot.rgb
        
        return None
    
    def extract_text_from_image(self, image_bytes: bytes) -> Optional[str]:
        """
        Extract text from captured screen using OCR.
        Useful for understanding text content without vision models.
        """
        if not TESSERACT_AVAILABLE or not PIL_AVAILABLE:
            logger.warning("OCR not available (install pytesseract and Pillow)")
            return None
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None
    
    def save_capture(self, image_bytes: bytes, 
                    context: Optional[str] = None) -> str:
        """
        Save a screen capture with metadata.
        Returns the filename for reference.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.png"
        filepath = self.storage_path / filename
        
        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Save metadata if provided
        if context:
            metadata_file = self.storage_path / f"capture_{timestamp}_metadata.json"
            import json
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "filename": filename,
                "ocr_text": self.extract_text_from_image(image_bytes)
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return filename
    
    def start_monitoring(self, 
                        callback: Callable[[bytes, str], None],
                        interval: int = 5,
                        detect_changes: bool = True):
        """
        Start monitoring screen changes.
        Calls callback with screen data when changes detected.
        
        Args:
            callback: Function to call with (image_bytes, change_description)
            interval: Seconds between checks
            detect_changes: Only trigger on significant changes
        """
        if self.monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(callback, interval, detect_changes)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info(f"Started screen monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop screen monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped screen monitoring")
    
    def _monitor_loop(self, callback: Callable, interval: int, detect_changes: bool):
        """Main monitoring loop"""
        last_screenshot = None
        
        while self.monitoring:
            try:
                # Capture current screen
                screenshot = self.capture_screen()
                if screenshot:
                    # Detect changes if requested
                    if detect_changes and last_screenshot:
                        if self._detect_significant_change(last_screenshot, screenshot):
                            callback(screenshot, "Significant screen change detected")
                    else:
                        callback(screenshot, "Regular screen capture")
                    
                    last_screenshot = screenshot
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
            
            time.sleep(interval)
    
    def _detect_significant_change(self, img1_bytes: bytes, img2_bytes: bytes) -> bool:
        """
        Detect if there's a significant change between two screenshots.
        Simple implementation - can be enhanced with better algorithms.
        """
        if not PIL_AVAILABLE:
            return True  # Assume change if we can't check
        
        try:
            img1 = Image.open(io.BytesIO(img1_bytes))
            img2 = Image.open(io.BytesIO(img2_bytes))
            
            # Resize for faster comparison
            size = (200, 150)
            img1_small = img1.resize(size)
            img2_small = img2.resize(size)
            
            # Simple pixel difference
            import numpy as np
            arr1 = np.array(img1_small)
            arr2 = np.array(img2_small)
            
            # Calculate difference
            diff = np.sum(np.abs(arr1 - arr2))
            total_pixels = size[0] * size[1] * 3  # RGB channels
            
            # If more than 10% pixels changed significantly
            change_ratio = diff / (total_pixels * 255)
            return change_ratio > 0.1
            
        except Exception as e:
            logger.error(f"Error detecting change: {e}")
            return True
    
    def _capture_screen_macos(self) -> Optional[bytes]:
        """macOS-specific screen capture using Quartz"""
        if not MACOS_AVAILABLE:
            return None
        
        try:
            # Create screenshot
            region = Quartz.CGRectInfinite
            image_ref = Quartz.CGWindowListCreateImage(
                region,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault
            )
            
            # Convert to data
            ns_image = AppKit.NSImage.alloc().initWithCGImage_size_(
                image_ref, AppKit.NSZeroSize
            )
            tiff_data = ns_image.TIFFRepresentation()
            bitmap_image = AppKit.NSBitmapImageRep.alloc().initWithData_(tiff_data)
            png_data = bitmap_image.representationUsingType_properties_(
                AppKit.NSPNGFileType, None
            )
            
            return bytes(png_data)
            
        except Exception as e:
            logger.error(f"macOS screen capture failed: {e}")
            return None
    
    def prepare_for_vision_model(self, image_bytes: bytes) -> str:
        """
        Prepare image data for vision model input.
        Most models expect base64 encoded strings.
        """
        return base64.b64encode(image_bytes).decode('utf-8')


class ScreenContextAnalyzer:
    """
    Analyzes screen content to provide context for ADAM.
    Helps understand what the user is working on.
    """
    
    def __init__(self, capture_service: ScreenCaptureService):
        self.capture_service = capture_service
    
    def analyze_screen_context(self, image_bytes: bytes) -> dict:
        """
        Analyze screen capture to extract context.
        Returns structured data about what's visible.
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "has_text": False,
            "extracted_text": None,
            "image_size": len(image_bytes),
            "suggestions": []
        }
        
        # Try OCR
        text = self.capture_service.extract_text_from_image(image_bytes)
        if text:
            context["has_text"] = True
            context["extracted_text"] = text
            
            # Analyze text for context
            if "error" in text.lower() or "exception" in text.lower():
                context["suggestions"].append("Detected error message - offer help?")
            if "def " in text or "class " in text or "function" in text:
                context["suggestions"].append("Code visible - provide coding assistance?")
        
        return context