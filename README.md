# AI Fitting Room

A local virtual try-on web app built with Python (Flask) and
[FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5). Everything runs on your own computer — no cloud API, no per-image cost.

- **Wardrobe:** upload clothing photos by gender and category
  - Men: Tops · Shorts · Trousers
  - Women: Tops · Shorts · Trousers · Skirts · Dresses
- **Try On:** upload a person photo, pick gender and clothes, and generate an image of the person wearing them.

## Install (once)

Needs internet and ~6 GB (Mac) / ~8 GB (Windows) of free disk. Takes 15–40 minutes, mostly downloads.

| | Mac (Apple Silicon recommended) | Windows 10/11 |
|---|---|---|
| Install | double-click `1-install-mac.command` | double-click `1-install-windows.bat` |
| Start | double-click `2-start-mac.command` | double-click `2-start-windows.bat` |

- Mac: if macOS refuses to open the file, right-click it → **Open** → **Open**.
- Windows: if you see "Windows protected your PC", click **More info** → **Run anyway**.
  With an NVIDIA (RTX) GPU the installer picks the CUDA build of PyTorch automatically — update your NVIDIA driver first. Without one it runs on CPU (several minutes per image).

The installer downloads Python 3.11, the libraries, the FASHN VTON code and the model weights (~2.3 GB). These are git-ignored, so a fresh clone just needs the installer.

## Use

1. Start the app — the browser opens `http://127.0.0.1:8765`.
2. Wait until the top bar shows **● AI ready** (the first load takes about a minute).
3. **① Wardrobe:** add clothing photos.
4. **② Try On:** upload a person photo → choose gender → choose clothes → **Generate try-on**.
5. To stop, close the terminal window.

## Folders

```
wardrobe/male/{top,shorts,pants}/                clothing photos (you can drop .jpg/.png files here and refresh)
wardrobe/female/{top,shorts,pants,skirt,dress}/
results/                                         generated images
```

## Tips

- Person photo: full body, facing the camera, good light, clean background.
- Clothing photo: whole garment visible, plain background. Set "Clothing photo type" to match (flat-lay vs. worn by a model).
- Several items are applied one at a time (bottoms first, then the top), so time grows with the number of items.
- Too slow? Choose **Fast (20 steps)**.

## Troubleshooting

- Errors are printed in the terminal window.
- If the Apple GPU (MPS) fails, the app switches to CPU automatically (slower).
- Force CPU: add `export TRYON_DEVICE=cpu` (Mac) or `set TRYON_DEVICE=cpu` (Windows) to the start script.

## License

FASHN VTON v1.5 is Apache-2.0 (commercial use allowed). The bundled `fashn-human-parser` has its own license — see
[fashn-ai/fashn-human-parser](https://github.com/fashn-AI/fashn-human-parser).
