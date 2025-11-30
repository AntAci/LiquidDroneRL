# Guide: Finding and Using Realistic 3D Drone Models

## Model Format Recommendations

### Best Format: **GLTF/GLB** (Recommended)
- **GLTF** (`.gltf`) - Text-based, includes separate texture files
- **GLB** (`.glb`) - Binary version, single file with embedded textures
- **Why**: Modern, efficient, widely supported by Three.js, includes materials and textures
- **File size**: Usually smaller and more optimized than OBJ
- **Quality**: Supports PBR materials, animations, and more
- **Cinema 4D**: Native export support via `File > Export > glTF Exporter` (choose GLB for best results)

### Alternative: **OBJ** (Works but older)
- **OBJ** (`.obj`) - Simple, widely available
- **Why**: Easy to find, simple format, works with Cinema 4D
- **Cinema 4D**: Export via `File > Export > Wavefront OBJ`
- **Limitations**: No embedded materials, larger file sizes, less efficient

### Avoid: **FBX, 3DS, DAE**
- These formats require additional loaders and are less web-friendly

## Where to Find Drone Models

### Free Resources:

1. **Sketchfab** (https://sketchfab.com)
   - Search: "quadcopter", "drone", "FPV drone"
   - Filter: Free downloads, CC0/CC-BY licenses
   - Formats: Usually GLTF/GLB available
   - Examples:
     - "DJI Phantom" models
     - "Racing Drone" models
     - "Quadcopter" models

2. **Poly Haven** (https://polyhaven.com/models)
   - High-quality free models
   - CC0 license (public domain)
   - GLTF format available

3. **TurboSquid Free** (https://www.turbosquid.com/Search/3D-Models/free)
   - Search: "drone", "quadcopter"
   - Filter: Free models
   - Check license before use

4. **CGTrader Free** (https://www.cgtrader.com/free-3d-models)
   - Search: "drone", "quadcopter"
   - Various formats available

5. **Thingiverse** (https://www.thingiverse.com)
   - 3D printing models, but many can be converted
   - Search: "drone", "quadcopter"

### Paid Resources (Higher Quality):

1. **Sketchfab Store**
2. **TurboSquid**
3. **CGTrader**

## Model Specifications

### What to Look For:

1. **Type**: Quadcopter (4 propellers) - matches your simulation
2. **Scale**: 
   - Model should be reasonably sized (not too large/small)
   - You can scale it in code (currently using scale: 1.5)
   - Typical drone size: ~0.2-0.5 meters in real world
3. **Complexity**:
   - Low to medium poly count (1000-5000 triangles) for web performance
   - Avoid ultra-high poly models (100k+ triangles) - they'll be slow
4. **Orientation**:
   - Model should face forward (usually +X or +Z in Three.js)
   - If wrong, you can rotate it in code
5. **Materials**:
   - GLTF models with materials/textures look best
   - OBJ models may need manual material assignment

## Recommended Model Types

Based on your simulation (2D movement with wind), these work well:

1. **Racing Drone** - Small, agile, matches fast movement
2. **DJI-style Quadcopter** - Professional look, recognizable
3. **FPV Drone** - Compact, good for visualization
4. **Simple Quadcopter** - Low poly, good performance

## Integration Steps

1. **Download a model** (GLTF/GLB preferred)
2. **Place it in** `demo/assets/drone.gltf` or `demo/assets/drone.glb`
3. **Update the HTML** - The code will automatically load it
4. **Adjust scale** if needed in `scene_config.json` or code

## File Size Considerations

- **Target**: < 5MB for web performance
- **GLTF/GLB**: Usually 1-3MB for good quality models
- **OBJ**: Can be larger (5-20MB) but still acceptable
- **If too large**: Use a model optimizer like:
  - gltf-pipeline (https://github.com/CesiumGS/gltf-pipeline)
  - Blender (export with compression)

## License Check

**Always check the license** before using a model:
- **CC0**: Public domain, use freely
- **CC-BY**: Attribution required (credit the author)
- **Commercial**: May require purchase for commercial use
- **Editorial use only**: Check restrictions

## Example Search Terms

- "quadcopter gltf"
- "drone 3d model free"
- "fpv racing drone glb"
- "dji phantom 3d model"
- "quadcopter low poly"

## Troubleshooting

### Model too large/small:
- Adjust scale in `scene_config.json` or in the HTML code
- Typical scale range: 0.5 to 3.0

### Model facing wrong direction:
- Rotate in code: `drone.rotation.y = Math.PI / 2` (or adjust as needed)

### Model not loading:
- Check browser console for errors
- Ensure file path is correct
- Check file format is supported (GLTF/GLB/OBJ)

### Performance issues:
- Use a lower poly model
- Optimize textures
- Use GLB instead of GLTF (single file, faster loading)

