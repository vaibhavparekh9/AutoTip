# AutoTip# AutoTip

A real-time pen-tip tracking and digital drawing using an Intel RealSense depth camera.

## Implementation
- **Optical-flow pen tracking** — Lucas-Kanade sparse flow tracks the pen tip at camera frame rate after a single click to initialise.
- **AprilTag boundary** — Two or more `tagStandard41h12` tags define the valid drawing region. Only movement inside the boundary is recorded.
- **Depth-aware pen up/down** — Aligned depth frames detect when the pen lifts off the surface (beyond a configurable threshold), automatically separating strokes.

## Hardware
- An Intel RealSense D400-series camera that supports aligned depth + colour streams (D435 recommended)
- Two or more AprilTags (from `tagStandard41h12` family)

## Setup
```bash
pip install opencv-python pyrealsense2 pupil-apriltags numpy
```
## Usage
1. Connect the RealSense D435 via USB
2. Place two AprilTags diagonally on your drawing surface
3. Run:
```bash
python autotip.py
```
4. Click on the pen tip in the video feed to start tracking
5. Press `R` to begin recording

### Controls

| Key | Action |
|-----|--------|
| **Click** | Set pen-tip position |
| **R** | Toggle recording |
| **E** | Toggle eraser |
| **C** | Clear canvas |
| **S** | Save drawing |
| **Q / Esc** | Quit |