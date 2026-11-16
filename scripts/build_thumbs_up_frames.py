"""Turn the thumbs-up GIF into the sprite sheet the site plays.

    python scripts/build_thumbs_up_frames.py

Reads data/thumbs-up-source.gif (16 hand-made frames, 380 x 252, on a flat
backdrop) and writes frontend/public/thumbs-up/frames.webp: the same 16 frames
side by side, at twice the size, on a transparent background.
frontend/src/components/ThumbsUp.jsx steps through them.

What is done to each frame, and why:

1. Backdrop removed. The figure is outlined in near-black wherever it meets the
   backdrop, so an edge pixel is outline at partial coverage: its alpha comes
   from how far its colour has moved from the backdrop, and its colour is set
   to the outline's, so no pale fringe survives onto the site's dark page.
2. Enlarged. The GIF is small; the work is done at 4x and the result saved at
   2x, which is enough for a high-density screen at the size it is shown.
3. First frame, hand on the hip. In the source that hand is drawn open with its
   fingers hanging; it is replaced by the closed fist of the other hand, so the
   opening pose is two fists on the hips.
4. Wink. From the frame where the thumb is up, for WINK_FRAMES, the eye on the
   viewer's right is closed: the skin is the frame's own, inpainted across the
   eye, with a closed-eye arc drawn on it.

5. Sparkle. The little strokes beside the thumb are black in the source, which
   would vanish on the site's dark page; they are recoloured to the suit's
   yellow.

6. Eyebrow. In the source the eyebrow on the viewer's left is drawn at full
   weight in the first frames and as a hairline in the rest, so the two brows
   stop matching. Wherever it has thinned, it is replaced by the first frame's,
   moved by however far the head has moved (measured on the other brow, which
   keeps its shape throughout).

7. Fist on the hip. In the source it is a plain blob with a single mark on it,
   which does not read as a hand. The mark is healed away and the fist is
   given what the thumbs-up hand in the same drawing has: fingers, as stacked
   rounded bars curled in against the body.

Needs opencv-python-headless, numpy and Pillow. None is a runtime dependency of
the site.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "thumbs-up-source.gif"
OUT = ROOT / "frontend" / "public" / "thumbs-up" / "frames.webp"

WORK, SAVE = 4, 2             # scale the work is done at, and the scale saved
CROP_X = (46, 314)            # columns of the source that the figure ever occupies
OUTLINE = (22, 20, 18)        # BGR
SPARKLE = (111, 208, 244)     # BGR: the yellow of the suit
EYE = (212.5, 77.0)           # the eye on the viewer's right, roughly
WINK_FRAMES = range(6, 13)
BROW_LEFT, BROW_RIGHT = (176.3, 58.1), (218.4, 59.6)  # centres in the first frame
# First-frame hand fix, in source pixels.
FIST_BOX = (240, 216, 292, 252)   # the closed fist on the other hip
FIST_SEED = (254, 240)
FIST_SHIFT = (-147, -3)           # where it has to move to: (dx, dy)
OPEN_HAND_BOX = (86, 218, 144, 252)
OPEN_HAND_SEED = (115, 232)


def read_frames() -> list[np.ndarray]:
    gif = Image.open(SOURCE)
    frames = []
    for index in range(gif.n_frames):
        gif.seek(index)
        frames.append(np.array(gif.convert("RGB"))[:, :, ::-1].copy())
    return frames


def cut_out(frame: np.ndarray) -> np.ndarray:
    """BGRA at WORK scale, backdrop transparent, edges matted."""
    backdrop = np.median(
        np.concatenate([frame[:5].reshape(-1, 3), frame[:, :5].reshape(-1, 3), frame[:, -5:].reshape(-1, 3)]),
        axis=0,
    )
    distance = np.linalg.norm(frame.astype(np.float32) - backdrop, axis=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((distance < 48).astype(np.uint8), connectivity=4)
    outside = np.zeros(frame.shape[:2], np.float32)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 25:  # the border, and the gap inside the bent arm
            outside[labels == index] = 1.0

    big = cv2.resize(frame, None, fx=WORK, fy=WORK, interpolation=cv2.INTER_LANCZOS4)
    big = cv2.bilateralFilter(big, 9, 28, 9)
    outside_big = cv2.resize(outside, None, fx=WORK, fy=WORK, interpolation=cv2.INTER_CUBIC) > 0.5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WORK + 1, 2 * WORK + 1))
    near_edge = cv2.dilate(outside_big.astype(np.uint8), kernel).astype(bool)
    deep_outside = cv2.erode(outside_big.astype(np.uint8), kernel).astype(bool)

    span = float(np.linalg.norm(backdrop - np.array(OUTLINE, np.float32)))
    ramp = np.clip(np.linalg.norm(big.astype(np.float32) - backdrop, axis=2) / (0.7 * span), 0.0, 1.0)
    alpha = np.where(deep_outside, 0.0, np.where(near_edge, ramp, 1.0)).astype(np.float32)

    colour = big.astype(np.float32)
    colour[(alpha < 0.999)] = OUTLINE
    return np.dstack([colour, alpha * 255.0])


def region(rgba: np.ndarray, box, seed, grow: float = 3.5) -> np.ndarray:
    """The hand at `seed`: its skin inside `box`, plus the outline within `grow` source pixels of it."""
    x0, y0, x1, y1 = (v * WORK for v in box)
    b, g, r, a = (rgba[..., i] for i in range(4))
    skin = (a > 128) & (r > 190) & (g > 140) & (g < 235) & (b > 90) & (r > b + 30)
    window = np.zeros(skin.shape, np.uint8)
    window[y0:y1, x0:x1] = skin[y0:y1, x0:x1]
    # Finger lines and the mark on the fist split the skin; close over them first.
    stroke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * WORK + 1, 3 * WORK + 1))
    window = cv2.morphologyEx(window, cv2.MORPH_CLOSE, stroke)
    _, labels = cv2.connectedComponents(window, connectivity=4)
    label = labels[seed[1] * WORK, seed[0] * WORK]
    if not label:
        return np.zeros(skin.shape, bool)
    reach = round(2 * grow * WORK) + 1
    outline = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
    grown = cv2.dilate((labels == label).astype(np.uint8), outline).astype(bool)
    dark = (b + g + r) < 200
    return (labels == label) | (grown & dark & (a > 128))


def filled(mask: np.ndarray, close: float = 0.0) -> np.ndarray:
    """`mask` with its holes filled; `close` (source pixels) first bridges gaps that wide."""
    solid = mask.astype(np.uint8)
    if close:
        size = round(2 * close * WORK) + 1
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(solid.shape, np.uint8)
    cv2.drawContours(out, contours, -1, 1, cv2.FILLED)
    return out.astype(bool)


def redraw_fist(rgba: np.ndarray) -> np.ndarray:
    """Turn the blob on the hip into a fist with fingers. The shape and its outline are kept."""
    x0, y0, x1, y1 = (v * WORK for v in FIST_BOX)
    b, g, r, a = (rgba[..., i] for i in range(4))
    skin = (a > 128) & (r > 190) & (g > 140) & (g < 235) & (b > 90) & (r > b + 30)
    window = np.zeros(skin.shape, np.uint8)
    window[y0:y1, x0:x1] = skin[y0:y1, x0:x1]
    stroke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * WORK + 1, 3 * WORK + 1))
    closed = cv2.morphologyEx(window, cv2.MORPH_CLOSE, stroke)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=4)
    if count < 2:
        return rgba
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hand = filled(labels == label).astype(np.uint8)   # the mark is a hole in the skin
    left, top, width, _ = (float(v) for v in stats[label, :4])

    out = rgba.copy()
    # The mark: whatever is dark well inside the hand.
    inside = cv2.erode(hand, stroke).astype(bool)
    mark = (inside & ((b + g + r) < 420)).astype(np.uint8)
    mark = cv2.dilate(mark, np.ones((3, 3), np.uint8), iterations=2 * WORK) & inside.astype(np.uint8)
    out[..., :3] = cv2.inpaint(out[..., :3].round().astype(np.uint8), mark, 3 * WORK, cv2.INPAINT_TELEA)

    # Fingers: bars from the body side of the fist to a little past its middle,
    # each closed by a rounded tip, as on the thumbs-up hand.
    ink = np.zeros(hand.shape, np.uint8)
    weight = round(1.8 * WORK)
    finger = 9.2 * WORK                 # height of one finger
    tip_x = left + 0.50 * width         # where the fingertips are
    first = top + 2.0 * WORK            # top of the first finger, just under the outline
    for index in range(3):
        y_top = first + index * finger
        centre = (round(tip_x), round(y_top + finger / 2))
        cv2.ellipse(ink, centre, (round(3.6 * WORK), round(finger / 2)), 0, -90, 90, 255, weight, cv2.LINE_AA)
        if index:
            # A finger's edge, bowed a little so it is not a ruled line.
            edge = np.array([[left, y_top - 0.5 * WORK], [(left + tip_x) / 2, y_top + 0.5 * WORK], [tip_x, y_top]])
            curve = [((1 - t) ** 2 * edge[0] + 2 * (1 - t) * t * edge[1] + t * t * edge[2]) for t in np.linspace(0, 1, 12)]
            cv2.polylines(ink, [np.round(curve).astype(np.int32)], False, 255, weight, cv2.LINE_AA)
    coverage = (ink.astype(np.float32) / 255.0 * hand)[..., None]
    out[..., :3] = out[..., :3] * (1 - coverage) + np.array(OUTLINE, np.float32) * coverage
    return out


def fix_hip_hand(rgba: np.ndarray) -> np.ndarray:
    """Replace the open hand of the first frame with a copy of the closed fist."""
    fist = region(rgba, FIST_BOX, FIST_SEED)
    out = rgba.copy()
    # The open hand is drawn with a heavier, doubled outline: reach further for
    # it. It is blacked out rather than cut away: the fist does not cover quite
    # the same area, and what is left over then reads as outline between cuff
    # and fist instead of as a hole.
    out[filled(region(rgba, OPEN_HAND_BOX, OPEN_HAND_SEED, grow=5.5), close=4)] = (*OUTLINE, 255)
    dx, dy = (v * WORK for v in FIST_SHIFT)
    ys, xs = np.nonzero(fist)
    out[ys + dy, xs + dx] = rgba[ys, xs]
    return out


def add_wink(rgba: np.ndarray) -> np.ndarray:
    b, g, r, a = (rgba[..., i] for i in range(4))
    ys, xs = np.mgrid[0 : rgba.shape[0], 0 : rgba.shape[1]]
    dark = ((b + g + r) < 170) & (a > 128) & (np.hypot(xs - EYE[0] * WORK, ys - EYE[1] * WORK) < 9 * WORK)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    if count < 2:
        return rgba
    eye_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centres[eye_label]
    eye = cv2.dilate((labels == eye_label).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2 * WORK)
    colour = cv2.inpaint(rgba[..., :3].round().astype(np.uint8), eye, 3 * WORK, cv2.INPAINT_TELEA).astype(np.float32)

    arc = np.zeros(eye.shape, np.uint8)
    cv2.ellipse(arc, (round(cx), round(cy + 5.5 * WORK)), (round(5.4 * WORK), round(5.2 * WORK)), 0, 205, 335,
                255, round(2.5 * WORK), cv2.LINE_AA)
    coverage = (arc.astype(np.float32) / 255.0)[..., None]
    colour = colour * (1 - coverage) + np.array(OUTLINE, np.float32) * coverage
    out = rgba.copy()
    out[..., :3] = colour
    return out


def light_sparkles(rgba: np.ndarray) -> np.ndarray:
    """Recolour anything detached from the figure: only the sparkle strokes are."""
    solid = (rgba[..., 3] > 40).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if count <= 2:
        return rgba
    figure = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = rgba.copy()
    out[(labels != figure) & (labels != 0), :3] = SPARKLE
    return out


def brow(rgba: np.ndarray, near) -> tuple[np.ndarray, np.ndarray, float]:
    """The eyebrow nearest `near` (source pixels): its mask, centre and area at WORK scale."""
    b, g, r, a = (rgba[..., i] for i in range(4))
    dark = ((b + g + r) < 260) & (a > 128)
    x0, x1 = round((near[0] - 12) * WORK), round((near[0] + 12) * WORK)
    y0, y1 = round((near[1] - 9) * WORK), round((near[1] + 9) * WORK)
    window = np.zeros(dark.shape, np.uint8)
    window[y0:y1, x0:x1] = dark[y0:y1, x0:x1]
    count, labels, stats, centres = cv2.connectedComponentsWithStats(window, connectivity=8)
    best, best_gap = 0, np.inf
    for index in range(1, count):
        # Not the hair: that component runs into the top or side of the window.
        left, top, width, height = stats[index, :4]
        if stats[index, cv2.CC_STAT_AREA] < 6 * WORK * WORK or left <= x0 or top <= y0 or left + width >= x1:
            continue
        gap = float(np.hypot(*(centres[index] / WORK - near)))
        if gap < best_gap:
            best, best_gap = index, gap
    if not best:
        return np.zeros(dark.shape, bool), np.array(near) * WORK, 0.0
    return labels == best, centres[best], float(stats[best, cv2.CC_STAT_AREA])


def match_brows(rgba: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Give this frame the first frame's left eyebrow if its own has thinned."""
    reference, _, reference_area = brow(first, BROW_LEFT)
    _, right_then, _ = brow(first, BROW_RIGHT)
    _, right_now, _ = brow(rgba, BROW_RIGHT)
    dx, dy = (right_now - right_then).round().astype(int)
    here = (BROW_LEFT[0] + dx / WORK, BROW_LEFT[1] + dy / WORK)
    thin, _, area = brow(rgba, here)
    if area >= 0.75 * reference_area:
        return rgba

    grow = np.ones((3, 3), np.uint8)
    out = rgba.copy()
    if thin.any():  # heal the hairline first, so none of it shows beside the new brow
        hole = cv2.dilate(thin.astype(np.uint8), grow, iterations=WORK)
        out[..., :3] = cv2.inpaint(out[..., :3].round().astype(np.uint8), hole, 2 * WORK, cv2.INPAINT_TELEA)
    patch = cv2.dilate(reference.astype(np.uint8), grow, iterations=WORK).astype(bool)
    ys, xs = np.nonzero(patch)
    out[ys + dy, xs + dx, :3] = first[ys, xs, :3]
    return out


def shrink(rgba: np.ndarray) -> np.ndarray:
    """WORK scale down to SAVE scale, averaging in premultiplied alpha so edges stay clean."""
    alpha = rgba[..., 3:4] / 255.0
    premultiplied = np.dstack([rgba[..., :3] * alpha, rgba[..., 3:4]])
    factor = SAVE / WORK
    small = cv2.resize(premultiplied, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    a = small[..., 3:4] / 255.0
    return np.dstack([np.where(a > 0, small[..., :3] / np.maximum(a, 1e-6), 0), small[..., 3:4]])


def build() -> np.ndarray:
    frames = []
    first = None
    for index, frame in enumerate(read_frames()):
        rgba = cut_out(frame)
        if index == 0:
            first = rgba       # kept as drawn: the reference for the eyebrow
        rgba = redraw_fist(rgba)
        if index == 0:
            rgba = fix_hip_hand(rgba)
        else:
            rgba = match_brows(rgba, first)
        if index in WINK_FRAMES:
            rgba = add_wink(rgba)
        small = shrink(light_sparkles(rgba))
        frames.append(small[:, CROP_X[0] * SAVE : CROP_X[1] * SAVE])
    return np.hstack(frames).round().clip(0, 255).astype(np.uint8)


def main() -> None:
    sheet = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), sheet, [cv2.IMWRITE_WEBP_QUALITY, 86])
    frames = sheet.shape[1] // ((CROP_X[1] - CROP_X[0]) * SAVE)
    print(f"wrote {OUT.relative_to(ROOT)}: {frames} frames of "
          f"{sheet.shape[1] // frames} x {sheet.shape[0]}, {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
