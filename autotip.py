from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pupil_apriltags as at
import pyrealsense2 as rs

Z_THRESHOLD = 0.05  # depth tolerance for pen-down detection (metres)
CAM_FPS = 30
CAM_WIDTH = 640
CAM_HEIGHT = 480
ERASER_RADIUS = 15  
MAX_LOST_FRAMES = 50  
WARMUP_SECONDS = 1.0  # to discard frames while sensor stabilises


class AutoTip:
    """Augmented-reality pen-tip tracker with depth-aware drawing."""

    def __init__(
        self,
        z_threshold: float = Z_THRESHOLD,
        cam_fps: int = CAM_FPS,
        cam_width: int = CAM_WIDTH,
        cam_height: int = CAM_HEIGHT,
        eraser_radius: int = ERASER_RADIUS,
        max_lost_frames: int = MAX_LOST_FRAMES,
        warmup_seconds: float = WARMUP_SECONDS,
    ):
        self.z_threshold = z_threshold
        self.cam_fps = cam_fps
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.eraser_radius = eraser_radius
        self.max_lost_frames = max_lost_frames
        self.warmup_seconds = warmup_seconds

        # Tracking state
        self._tip: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._lost_frames = 0
        self._baseline_depth: float | None = None
        self._pen_down = False

        # Drawing state
        self._strokes: list[list[tuple[int, int]]] = []
        self._active: list[tuple[int, int]] = []
        self._recording = False
        self._eraser = False

        # Bounding rectangle from AprilTag corners
        self._bounds: tuple[float, float, float, float] | None = None

        self._canvas: np.ndarray | None = None
        self._canvas_dirty = False

        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._detector: at.Detector | None = None


    def _init_realsense(self) -> None:
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.depth, self.cam_width, self.cam_height,
            rs.format.z16, self.cam_fps,
        )
        cfg.enable_stream(
            rs.stream.color, self.cam_width, self.cam_height,
            rs.format.bgr8, self.cam_fps,
        )
        self._pipeline.start(cfg)
        # align depth pixels to the colour coordinate space 
        self._align = rs.align(rs.stream.color)

    def _init_detector(self) -> None:
        self._detector = at.Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    # Mouse callback 
    def _on_click(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        """Initialise pen-tip position on left-click."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self._tip = np.array([[x, y]], dtype=np.float32)
        self._lost_frames = 0
        self._baseline_depth = None
        self._pen_down = False
        self._commit_stroke()
        print(f"Tip set at ({x}, {y})")

    # AprilTag boundary 
    def _update_boundary(self, gray: np.ndarray) -> None:
        tags = self._detector.detect(gray)
        if not tags:
            return
        corners = [c for tag in tags for c in tag.corners]
        if len(corners) < 4:
            return
        xs, ys = zip(*corners)
        self._bounds = (min(xs), min(ys), max(xs), max(ys))

    def _in_bounds(self, x: float, y: float) -> bool:
        if self._bounds is None:
            return False
        x0, y0, x1, y1 = self._bounds
        return x0 <= x <= x1 and y0 <= y <= y1

    # Depth-based pen detection
    def _sample_depth(
        self, depth_frame: rs.depth_frame, x: float, y: float,
    ) -> float | None:
        """Return depth in metres at pixel (x, y), or None if invalid."""
        ix = max(0, min(int(round(x)), self.cam_width - 1))
        iy = max(0, min(int(round(y)), self.cam_height - 1))
        d = depth_frame.get_distance(ix, iy)
        return d if d > 0.0 else None

    def _update_pen_state(
        self, depth_frame: rs.depth_frame, x: float, y: float,
    ) -> None:
        """Compare current depth against baseline to decide pen-down."""
        depth = self._sample_depth(depth_frame, x, y)
        if depth is None:
            return  # retain previous state on noisy/invalid reading

        if self._baseline_depth is None:
            self._baseline_depth = depth
            self._pen_down = True
            return

        self._pen_down = abs(depth - self._baseline_depth) < self.z_threshold

    # Optical flow tracking 
    def _track(self, gray: np.ndarray) -> tuple[float, float] | None:
        """Advance the tip position by one frame using Lucas-Kanade flow."""
        if self._tip is None or self._prev_gray is None:
            return None

        new, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._tip, None,
        )
        if status[0][0] == 1:
            self._tip = new
            self._lost_frames = 0
            return float(new[0][0]), float(new[0][1])

        self._lost_frames += 1
        if self._lost_frames >= self.max_lost_frames:
            print("Tracking lost — click to reinitialise.")
            self._tip = None
            self._commit_stroke()
        return None

    # Stroke management 
    def _commit_stroke(self) -> None:
        """Promote the active stroke to the committed list (if non-trivial)."""
        if len(self._active) >= 2:
            self._strokes.append(self._active)
        self._active = []

    def _add_point(self, x: float, y: float) -> None:
        """Append a point and incrementally draw the new segment on the canvas."""
        pt = (int(x), int(y))
        if self._active:
            cv2.line(self._canvas, self._active[-1], pt, (0, 255, 0), 2)
        self._active.append(pt)

    def _erase_at(self, x: int, y: int) -> None:
        """Remove points within eraser_radius, splitting strokes as needed."""
        r2 = self.eraser_radius ** 2
        all_strokes = self._strokes + ([self._active] if self._active else [])
        rebuilt: list[list[tuple[int, int]]] = []

        for stroke in all_strokes:
            seg: list[tuple[int, int]] = []
            for px, py in stroke:
                if (px - x) ** 2 + (py - y) ** 2 > r2:
                    seg.append((px, py))
                else:
                    if len(seg) >= 2:
                        rebuilt.append(seg)
                    seg = []
            if len(seg) >= 2:
                rebuilt.append(seg)

        self._strokes = rebuilt
        self._active = []
        self._canvas_dirty = True

    # Canvas rendering 
    def _rebuild_canvas(self) -> None:
        """Full redraw — only called after an erase or clear operation."""
        self._canvas[:] = 0
        for stroke in self._strokes:
            pts = np.array(stroke, dtype=np.int32)
            cv2.polylines(self._canvas, [pts], False, (0, 255, 0), 2)
        if len(self._active) >= 2:
            pts = np.array(self._active, dtype=np.int32)
            cv2.polylines(self._canvas, [pts], False, (0, 255, 0), 2)
        self._canvas_dirty = False

    # HUD overlay 
    def _draw_hud(self, frame: np.ndarray) -> None:
        parts = ["REC" if self._recording else "PAUSED"]
        if self._eraser:
            parts.append("ERASER")
        if self._tip is not None:
            parts.append("DOWN" if self._pen_down else "UP")

        cv2.putText(
            frame, " | ".join(parts), (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
        )

        if self._bounds:
            bx0, by0, bx1, by1 = self._bounds
            cv2.rectangle(
                frame, (int(bx0), int(by0)), (int(bx1), int(by1)),
                (255, 0, 0), 2,
            )

        if self._tip is not None:
            cx, cy = int(self._tip[0][0]), int(self._tip[0][1])
            colour = (0, 0, 255) if self._eraser else (0, 255, 0)
            arm = 10
            cv2.line(frame, (cx - arm, cy), (cx + arm, cy), colour, 1)
            cv2.line(frame, (cx, cy - arm), (cx, cy + arm), colour, 1)
            if self._eraser:
                cv2.circle(frame, (cx, cy), self.eraser_radius, (0, 0, 255), 1)

    # Save drawing 
    def _save(self, colour_image: np.ndarray) -> None:
        combined = cv2.addWeighted(colour_image, 0.7, self._canvas, 0.3, 0)
        name = f"drawing_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(os.getcwd(), name)
        cv2.imwrite(path, combined)
        print(f"Saved → {path}")

    # Main loop 
    def run(self) -> None:
        """Start the camera feed, tracking, and drawing loop."""
        self._init_realsense()
        self._init_detector()
        self._canvas = np.zeros(
            (self.cam_height, self.cam_width, 3), dtype=np.uint8,
        )

        win = "AutoTip"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._on_click)

        print(
            "\nControls:\n"
            "  Click   — set pen-tip position (place pen on surface first)\n"
            "  R       — toggle recording\n"
            "  E       — toggle eraser\n"
            "  C       — clear canvas\n"
            "  S       — save drawing\n"
            "  Q / Esc — quit\n"
        )

        t0 = time.time()

        try:
            while True:
                if time.time() - t0 < self.warmup_seconds:
                    self._pipeline.wait_for_frames()
                    continue

                frames = self._pipeline.wait_for_frames()
                aligned = self._align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                colour = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)

                self._update_boundary(gray)

                # Tracking+drawing 
                if self._tip is not None:
                    pos = self._track(gray)
                    if pos:
                        x, y = pos
                        self._update_pen_state(depth_frame, x, y)

                        should_draw = (
                            self._recording
                            and self._in_bounds(x, y)
                            and self._pen_down
                        )
                        if should_draw:
                            if self._eraser:
                                self._erase_at(int(x), int(y))
                            else:
                                self._add_point(x, y)
                        elif self._active:
                            self._commit_stroke()

                self._prev_gray = gray

                if self._canvas_dirty:
                    self._rebuild_canvas()

                # Display 
                display = cv2.addWeighted(colour, 0.7, self._canvas, 0.3, 0)
                self._draw_hud(display)
                cv2.imshow(win, display)

                # Keyboard inputs
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    self._recording = not self._recording
                    if self._recording:
                        print("Recording started.")
                    else:
                        self._commit_stroke()
                        print("Recording paused.")
                elif key == ord("e"):
                    self._eraser = not self._eraser
                    print(f"Eraser {'ON' if self._eraser else 'OFF'}.")
                elif key == ord("c"):
                    self._strokes.clear()
                    self._active.clear()
                    self._canvas_dirty = True
                    print("Canvas cleared.")
                elif key == ord("s"):
                    self._save(colour)

        finally:
            self._pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    AutoTip().run()