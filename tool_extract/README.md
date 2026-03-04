# DLux Tool Extract UI

Local toolchain for parsing an input SVG, building regions/zones, packing them, and previewing or exporting results. The UI is a React + Konva app, served by a Flask backend that runs the geometry/packing pipeline and exports SVG/PNG/GIF/PDF artifacts.

## What this project does
- Loads `convoi.svg`, extracts regions/zones, computes a packed layout, and writes debug/preview outputs.
- Provides an interactive web UI to inspect zones, labels, packing results, and export artifacts.
- Exposes a set of JSON APIs used by the UI to recompute, save state, and export files.
- Includes a legacy upload-based polygonize tool (`app.py`) that converts a JSON + image pair into SVG outputs.

## Tech stack
- Backend: Python, Flask
- Geometry/packing: OpenCV, NumPy, Shapely, rectpack, Pillow
- Frontend: React + Konva (Vite)

## Requirements
- Python 3.10+ recommended
- Node.js 18+ recommended

Python packages used (install into a venv):
- `flask`
- `numpy`
- `opencv-python`
- `shapely`
- `rectpack`
- `pillow`
- Optional for PDF export: `reportlab`, `svglib`

## Setup
1. Create and activate a venv.
2. Install Python deps:
   ```bash
   python -m pip install flask numpy opencv-python shapely rectpack pillow
   python -m pip install reportlab svglib  # optional, for PDF export
   ```
3. Install frontend deps:
   ```bash
   cd frontend
   npm install
   ```

## Run (recommended dev flow)
1. Start the backend API:
   ```bash
   python server.py
   ```
   The API listens on `http://127.0.0.1:5000`.
2. Start the frontend dev server:
   ```bash
   cd frontend
   npm run dev
   ```
   Vite runs on `http://127.0.0.1:5173` and proxies `/api` + `/out` to Flask.

## Run (single-server build)
1. Build the frontend:
   ```bash
   cd frontend
   npm run build
   ```
2. Start the backend:
   ```bash
   python server.py
   ```
   Flask will serve `frontend/dist`.

## Legacy upload tool
This repo also includes a simple upload flow (JSON + image) for polygonize + offset:
```bash
python app.py
```
Then open `http://127.0.0.1:8008`.

## Key outputs and files
- `convoi.svg`: primary input SVG
- `packed.svg`, `packed.png`, `packed_page2.svg`: packing results
- `zone_outline.svg`, `zone.svg`, `region.svg`: geometry outputs
- `export/`: exported SVG/HTML/PDF/GIF files
- `ui_state.json`, `scene_cache.json`: UI and scene cache

## API overview (Flask)
- `GET /api/scene`: compute scene, returns geometry/packing data
- `POST /api/render`: recompute pipeline with parameters
- `GET/POST /api/state`: load/save UI state
- `GET/POST /api/packed_labels`: load/save packed labels
- `POST /api/export`: export packed/zone SVGs
- `POST /api/export_pdf`: export PDF (needs reportlab + svglib)
- `POST /api/export_sim_video`: export GIF simulation
- `POST /api/save_svg`: write overlays/lines into `convoi.svg`
- `POST /api/save_html`: export HTML to `export/`
- `GET /api/download_*`: download exported files

## Configuration (common env vars)
These can be passed via environment variables or API payload/query params:
- `INTERSECT_SNAP`, `LINE_EXTEND`
- `PACK_PADDING`, `PACK_MARGIN_X`, `PACK_MARGIN_Y`, `PACK_BLEED`
- `PACK_GRID_STEP`, `PACK_ANGLE_STEP`, `PACK_MODE`
- `DRAW_SCALE`

## Notes
- Most geometry/packing logic lives in `scripts/` and is re-exported via `new_toy.py`.
- The backend precomputes and writes multiple debug assets to help validate packing.
