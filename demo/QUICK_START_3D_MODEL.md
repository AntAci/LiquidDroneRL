# Quick Start: Adding a Realistic 3D Drone Model

## Step 1: Download a Model

1. Go to **Sketchfab** (https://sketchfab.com) or another source (see `DRONE_MODEL_GUIDE.md`)
2. Search for: **"quadcopter"**, **"drone"**, or **"FPV drone"**
3. Filter: **Free downloads**, **CC0 or CC-BY license**
4. Download in **GLB format** (preferred) or **GLTF format**

## Step 2: Place the Model

Place your downloaded model file in the `demo/assets/` folder with one of these names:
- `drone.glb` (preferred - single binary file)
- `drone.gltf` (text-based, may have separate texture files)
- `drone.obj` (older format, works but less efficient)

**Example:**
```
demo/assets/drone.glb
```

## Step 3: Test It

1. Start the bridge: `python demo/bridge.py`
2. Open `demo/renderer_3d.html` in your browser
3. Run the demo: `python demo/run_3d_demo.py --model-path models/liquid_policy.zip`
4. The model should load automatically!

## Troubleshooting

### Model doesn't appear:
- Check browser console (F12) for errors
- Verify file is in `demo/assets/` folder
- Check file name matches exactly: `drone.glb`, `drone.gltf`, or `drone.obj`
- If using GLTF, ensure all texture files are in the same folder

### Model is too big/small:
- Edit `droneScale` in `renderer_3d.html` (around line 100)
- Current value: `1.5` - try `0.5` to `3.0` range

### Model facing wrong direction:
- Add rotation in `renderer_3d.html` after model loads:
  ```javascript
  drone.rotation.y = Math.PI / 2; // Rotate 90 degrees
  ```

### Model has no color/texture:
- OBJ files may need materials - the code applies a default blue material
- GLTF/GLB files should have materials included

## File Format Priority

The code tries to load models in this order:
1. `drone.glb` (best - single file, fast)
2. `drone.gltf` (good - may have separate textures)
3. `drone.obj` (works - older format)
4. Falls back to simple box if none found

## Recommended Models

Good search terms on Sketchfab:
- "quadcopter glb"
- "racing drone gltf"
- "fpv drone free"
- "dji phantom 3d model"

Look for models with:
- ✅ Low to medium poly count (1000-5000 triangles)
- ✅ File size < 5MB
- ✅ CC0 or CC-BY license
- ✅ GLB format available

## Current Setup

- **Model location**: `demo/assets/drone.*`
- **Scale**: 1.5x (adjustable in code)
- **Color**: Blue (#00aaff) - changes to red at boundaries
- **Shadows**: Enabled
- **Position**: Updates in real-time from simulation

The renderer automatically detects and loads your model - no code changes needed after placing the file!





