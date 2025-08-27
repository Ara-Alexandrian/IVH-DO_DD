"""
Cornerstone.js Medical Imaging Viewer for Streamlit
Professional-grade DICOM viewer with proper medical imaging controls
"""
import streamlit as st
import streamlit.components.v1 as components
import json
import numpy as np
import base64
from pathlib import Path

def create_cornerstone_viewer(ct_volume, current_slice=0, window_width=400, window_center=50, masks=None):
    """
    Create a Cornerstone.js medical imaging viewer
    
    Args:
        ct_volume: 3D numpy array of CT data in HU
        current_slice: Current slice index
        window_width: Window width for display
        window_center: Window level for display
        masks: Optional dictionary of mask overlays
    """
    
    if ct_volume is None:
        st.error("No CT volume loaded")
        return
    
    # Convert CT slice to proper format for Cornerstone
    slice_data = ct_volume[current_slice].astype(np.float32)
    
    # Debug info
    print(f"DEBUG: Cornerstone - CT slice shape: {slice_data.shape}")
    print(f"DEBUG: Cornerstone - HU range: {np.min(slice_data)} to {np.max(slice_data)}")
    print(f"DEBUG: Cornerstone - Sample values: {slice_data[256, 250:260]}")
    
    # For testing, let's try a smaller region first - center 128x128 patch
    center_y, center_x = slice_data.shape[0] // 2, slice_data.shape[1] // 2
    patch_size = 128
    y1, y2 = center_y - patch_size//2, center_y + patch_size//2  
    x1, x2 = center_x - patch_size//2, center_x + patch_size//2
    patch_data = slice_data[y1:y2, x1:x2].copy()
    
    print(f"DEBUG: Using {patch_size}x{patch_size} center patch for testing")
    print(f"DEBUG: Patch HU range: {np.min(patch_data)} to {np.max(patch_data)}")
    
    # Create simplified image data with small patch
    image_data = {
        'imageId': f'slice_{current_slice}',
        'rows': int(patch_data.shape[0]),
        'columns': int(patch_data.shape[1]), 
        'pixelData': patch_data.flatten().tolist(),
        'windowWidth': window_width,
        'windowCenter': window_center,
        'slope': 1.0,
        'intercept': 0.0,
        'minPixelValue': float(np.min(patch_data)),
        'maxPixelValue': float(np.max(patch_data)),
        'sliceIndex': current_slice,
        'totalSlices': ct_volume.shape[0],
        'dataLength': int(patch_data.size)
    }
    
    print(f"DEBUG: Cornerstone - Patch data length: {len(image_data['pixelData'])} pixels")
    print(f"DEBUG: Cornerstone - First 10 pixel values: {image_data['pixelData'][:10]}")
    
    # Cornerstone.js HTML template
    cornerstone_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Imaging Viewer</title>
        
        <!-- Cornerstone.js CSS -->
        <style>
            body {{
                margin: 0;
                padding: 20px;
                background-color: #1a1a1a;
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            
            #dicomImage {{
                width: 100%;
                height: 600px;
                background-color: #000;
                border: 2px solid #444;
                cursor: crosshair;
            }}
            
            .controls {{
                margin: 10px 0;
                display: flex;
                gap: 20px;
                align-items: center;
                flex-wrap: wrap;
            }}
            
            .control-group {{
                display: flex;
                align-items: center;
                gap: 10px;
                background-color: #2d2d2d;
                padding: 10px;
                border-radius: 5px;
            }}
            
            .slider {{
                width: 150px;
            }}
            
            .info {{
                background-color: #2d2d2d;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-family: monospace;
            }}
            
            .preset-btn {{
                padding: 8px 12px;
                background-color: #444;
                border: 1px solid #666;
                color: white;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            
            .preset-btn:hover {{
                background-color: #555;
            }}
            
            .preset-btn.active {{
                background-color: #0066cc;
            }}
        </style>
        
        <!-- Cornerstone.js Libraries -->
        <script src="https://cdn.jsdelivr.net/npm/cornerstone-core@2.6.1/dist/cornerstone.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/cornerstone-math@0.1.10/dist/cornerstoneMath.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/cornerstone-tools@6.0.10/dist/cornerstoneTools.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
    </head>
    <body>
        <div class="controls">
            <div class="control-group">
                <label>Window Width:</label>
                <input type="range" id="windowWidth" class="slider" min="1" max="4000" value="{window_width}" step="10">
                <span id="wwValue">{window_width}</span>
            </div>
            
            <div class="control-group">
                <label>Window Level:</label>
                <input type="range" id="windowLevel" class="slider" min="-1000" max="3000" value="{window_center}" step="10">
                <span id="wlValue">{window_center}</span>
            </div>
            
            <div class="control-group">
                <label>Slice:</label>
                <input type="range" id="sliceSlider" class="slider" min="0" max="{ct_volume.shape[0]-1}" value="{current_slice}" step="1">
                <span id="sliceValue">{current_slice + 1}/{ct_volume.shape[0]}</span>
            </div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>CT Presets:</label>
                <button class="preset-btn" onclick="applyPreset(350, 50)">Soft Tissue</button>
                <button class="preset-btn" onclick="applyPreset(2000, 500)">Bone</button>
                <button class="preset-btn" onclick="applyPreset(1500, -600)">Lung</button>
                <button class="preset-btn" onclick="applyPreset(4000, 1000)">Metal</button>
                <button class="preset-btn" onclick="applyPreset(80, 25)">Brain</button>
            </div>
        </div>
        
        <div id="dicomImage"></div>
        
        <div class="info">
            <div><strong>Image Info:</strong> {slice_data.shape[0]}×{slice_data.shape[1]} pixels</div>
            <div><strong>HU Range:</strong> {np.min(slice_data):.0f} to {np.max(slice_data):.0f}</div>
            <div><strong>Mouse:</strong> Left-drag = Window/Level | Right-drag = Zoom | Middle = Pan</div>
            <div id="pixelInfo"><strong>Pixel Value:</strong> Move mouse over image</div>
        </div>
        
        <script>
            // Initialize Cornerstone
            cornerstone.enable(document.getElementById('dicomImage'));
            
            // Image data from Python
            const imageData = {json.dumps(image_data)};
            
            // Simple direct array loader
            function loadImage(imageId) {{
                console.log('Loading image:', imageId);
                console.log('Image data - rows:', imageData.rows, 'columns:', imageData.columns);
                console.log('Pixel data length:', imageData.pixelData.length);
                console.log('Expected length:', imageData.rows * imageData.columns);
                console.log('Sample pixel values:', imageData.pixelData.slice(0, 10));
                console.log('Min/Max pixel values:', imageData.minPixelValue, imageData.maxPixelValue);
                console.log('Window/Level:', imageData.windowWidth, imageData.windowCenter);
                
                const promise = new Promise((resolve, reject) => {{
                    try {{
                        // Convert JavaScript array to Float32Array
                        const pixelData = new Float32Array(imageData.pixelData);
                        
                        console.log('Created Float32Array with length:', pixelData.length);
                        console.log('Float32Array sample:', Array.from(pixelData.slice(0, 10)));
                        
                        const image = {{
                            imageId: imageData.imageId,
                            minPixelValue: imageData.minPixelValue,
                            maxPixelValue: imageData.maxPixelValue,
                            slope: imageData.slope,
                            intercept: imageData.intercept,
                            windowCenter: imageData.windowCenter,
                            windowWidth: imageData.windowWidth,
                            rows: imageData.rows,
                            columns: imageData.columns,
                            height: imageData.rows,
                            width: imageData.columns,
                            color: false,
                            columnPixelSpacing: 1.0,
                            rowPixelSpacing: 1.0,
                            invert: false,
                            sizeInBytes: pixelData.length * 4,
                            getPixelData: function() {{
                                console.log('getPixelData called - returning Float32Array');
                                return pixelData;
                            }}
                        }};
                        
                        console.log('Image object created - dimensions:', image.width, 'x', image.height);
                        resolve(image);
                    }} catch (error) {{
                        console.error('Error in loadImage:', error);
                        reject(error);
                    }}
                }});
                return promise;
            }}
            
            // Register our custom loader
            cornerstone.registerImageLoader('custom', loadImage);
            
            // Display the image with error handling
            const element = document.getElementById('dicomImage');
            console.log('About to display image with ID:', 'custom:' + imageData.imageId);
            
            cornerstone.displayImage(element, 'custom:' + imageData.imageId)
                .then(() => {{
                    console.log('Image displayed successfully!');
                    const viewport = cornerstone.getViewport(element);
                    console.log('Current viewport:', viewport);
                }})
                .catch((error) => {{
                    console.error('Failed to display image:', error);
                    document.getElementById('dicomImage').innerHTML = 
                        '<div style="color: red; padding: 20px; text-align: center;">Failed to load image: ' + error.message + '</div>';
                }});
            
            // Enable tools
            cornerstoneTools.external.cornerstone = cornerstone;
            cornerstoneTools.external.cornerstoneMath = cornerstoneMath;
            cornerstoneTools.external.Hammer = Hammer;
            cornerstoneTools.init();
            
            // Add tools
            const WwwcTool = cornerstoneTools.WwwcTool;
            const PanTool = cornerstoneTools.PanTool;
            const ZoomTool = cornerstoneTools.ZoomTool;
            
            cornerstoneTools.addTool(WwwcTool);
            cornerstoneTools.addTool(PanTool);
            cornerstoneTools.addTool(ZoomTool);
            
            cornerstoneTools.setToolActive('Wwwc', {{ mouseButtonMask: 1 }}); // Left mouse
            cornerstoneTools.setToolActive('Pan', {{ mouseButtonMask: 4 }}); // Middle mouse
            cornerstoneTools.setToolActive('Zoom', {{ mouseButtonMask: 2 }}); // Right mouse
            
            // Update controls when image changes
            element.addEventListener('cornerstoneimagerendered', function(e) {{
                const viewport = cornerstone.getViewport(element);
                document.getElementById('windowWidth').value = viewport.voi.windowWidth;
                document.getElementById('windowLevel').value = viewport.voi.windowCenter;
                document.getElementById('wwValue').textContent = Math.round(viewport.voi.windowWidth);
                document.getElementById('wlValue').textContent = Math.round(viewport.voi.windowCenter);
            }});
            
            // Mouse move for pixel values
            element.addEventListener('mousemove', function(e) {{
                const rect = element.getBoundingClientRect();
                const x = Math.round(e.clientX - rect.left);
                const y = Math.round(e.clientY - rect.top);
                
                const enabledElement = cornerstone.getEnabledElement(element);
                if (enabledElement && enabledElement.image) {{
                    const pixelCoords = cornerstone.pageToPixel(element, x + rect.left, y + rect.top);
                    const px = Math.round(pixelCoords.x);
                    const py = Math.round(pixelCoords.y);
                    
                    if (px >= 0 && px < enabledElement.image.width && py >= 0 && py < enabledElement.image.height) {{
                        const pixelData = enabledElement.image.getPixelData();
                        const pixelValue = pixelData[py * enabledElement.image.width + px];
                        document.getElementById('pixelInfo').innerHTML = 
                            `<strong>Pixel Value:</strong> (${{px}}, ${{py}}) = ${{Math.round(pixelValue)}} HU`;
                    }}
                }}
            }});
            
            // Control handlers
            document.getElementById('windowWidth').addEventListener('input', function(e) {{
                const viewport = cornerstone.getViewport(element);
                viewport.voi.windowWidth = parseFloat(e.target.value);
                cornerstone.setViewport(element, viewport);
                document.getElementById('wwValue').textContent = e.target.value;
            }});
            
            document.getElementById('windowLevel').addEventListener('input', function(e) {{
                const viewport = cornerstone.getViewport(element);
                viewport.voi.windowCenter = parseFloat(e.target.value);
                cornerstone.setViewport(element, viewport);
                document.getElementById('wlValue').textContent = e.target.value;
            }});
            
            // Preset function
            function applyPreset(width, level) {{
                const viewport = cornerstone.getViewport(element);
                viewport.voi.windowWidth = width;
                viewport.voi.windowCenter = level;
                cornerstone.setViewport(element, viewport);
                
                document.getElementById('windowWidth').value = width;
                document.getElementById('windowLevel').value = level;
                document.getElementById('wwValue').textContent = width;
                document.getElementById('wlValue').textContent = level;
            }}
            
            // Fit to window
            cornerstone.resize(element);
            cornerstone.fitToWindow(element);
            
        </script>
    </body>
    </html>
    """
    
    # Display the Cornerstone viewer
    components.html(cornerstone_html, height=800, scrolling=False)
    
    return current_slice