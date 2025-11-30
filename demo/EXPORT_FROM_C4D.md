# Exporting Your Cinema 4D Drone Model

## Quick Export Steps

You have a `.c4d` file that needs to be exported to a web-compatible format. Here's how:

### Option 1: Export to GLB (Recommended - Best Quality)

1. **Open your C4D file**: `Drone by Versales Oliveira.c4d`
2. **Go to**: `File > Export > glTF Exporter...`
3. **In the export dialog**:
   - Choose **GLB** format (binary, single file)
   - Or choose **glTF** if you want separate texture files
4. **Export settings**:
   - ✅ Include materials/textures
   - ✅ Include animations (if any)
   - ✅ Embed textures (for GLB)
5. **Save as**: `drone.glb` in the `demo/assets/` folder
6. **Click Export**

### Option 2: Export to OBJ (Simpler, but less features)

1. **Open your C4D file**
2. **Go to**: `File > Export > Wavefront OBJ...`
3. **Save as**: `drone.obj` in the `demo/assets/` folder
   - ⚠️ Note: This will overwrite the existing simple cube `drone.obj`
4. **Export settings**:
   - ✅ Export materials (creates `.mtl` file)
   - ✅ Export textures
5. **Click Save**

## After Export

1. **Place the exported file** in `demo/assets/`:
   - `drone.glb` (preferred)
   - `drone.gltf` (alternative)
   - `drone.obj` (works, but less efficient)

2. **Test it**:
   - Start bridge: `python demo/bridge.py`
   - Open `demo/renderer_3d.html` in browser
   - Run demo: `python demo/run_3d_demo.py --model-path models/liquid_policy.zip`
   - Your model should appear!

## Format Comparison

| Format | Pros | Cons |
|--------|------|------|
| **GLB** | Single file, fast, includes materials | Requires C4D R20+ |
| **GLTF** | Includes materials, modern | May have separate texture files |
| **OBJ** | Simple, universal support | Larger files, may lose materials |

## Troubleshooting

### "glTF Exporter" not in menu?
- You need **Cinema 4D R20 or newer**
- For older versions, use OBJ export instead

### Model appears too large/small?
- Edit `droneScale` in `renderer_3d.html` (around line 100)
- Try values between `0.5` and `3.0`

### Model facing wrong direction?
- In C4D, rotate your model before exporting
- Or add rotation in code after loading

### Materials/textures missing?
- GLB/GLTF: Make sure "Include materials" is checked
- OBJ: May need to manually assign materials in code (already handled)

## Current Files

- ✅ `Drone by Versales Oliveira.c4d` - Your source file (needs export)
- ⚠️ `drone.obj` - Simple cube placeholder (will be replaced)

## Recommended Workflow

1. **Export to GLB** (if C4D R20+):
   ```
   File > Export > glTF Exporter > Choose GLB > Save as drone.glb
   ```

2. **Place in assets folder**:
   ```
   demo/assets/drone.glb
   ```

3. **Test immediately** - no code changes needed!

The renderer will automatically detect and load your GLB file.

