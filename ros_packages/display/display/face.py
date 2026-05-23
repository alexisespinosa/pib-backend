import math
import time
from dataclasses import dataclass, field

import pygame


SCREEN_W = 1024
SCREEN_H = 600

BG_COLOR = (0, 0, 0)
EYE_COLOR = (41, 171, 226)  # pib blue

# ── Emotion geometry ────────────────────────────────────────────────
# All coordinates are relative to SCREEN_W/SCREEN_H (0..1 normalized).
# Each emotion defines: left eye, right eye, left brow, right brow, mouth.


@dataclass
class EyeParams:
    cx: float  # center x (0..1)
    cy: float  # center y (0..1)
    rx: float  # horizontal radius (0..1)
    ry: float  # vertical radius (0..1)
    openness: float = 1.0  # 0 = closed, 1 = fully open (clips top/bottom)
    squint_top: float = 0.0  # extra top clipping (0..1)
    squint_bottom: float = 0.0  # extra bottom clipping


@dataclass
class BrowParams:
    cx: float = 0.0
    cy: float = 0.0
    width: float = 0.08
    angle: float = 0.0  # degrees, positive = inner end raised
    visible: bool = False


@dataclass
class MouthParams:
    cx: float = 0.5
    cy: float = 0.85
    width: float = 0.12
    height: float = 0.0  # 0 = line, positive = smile, negative = frown
    openness: float = 0.0  # 0 = closed, >0 = open mouth height
    visible: bool = False


@dataclass
class ExtraAnimation:
    kind: str = "none"  # "none", "thinking_dots", "thinking_spinner"


@dataclass
class EmotionState:
    left_eye: EyeParams
    right_eye: EyeParams
    left_brow: BrowParams = field(default_factory=BrowParams)
    right_brow: BrowParams = field(default_factory=BrowParams)
    mouth: MouthParams = field(default_factory=MouthParams)
    extra: ExtraAnimation = field(default_factory=ExtraAnimation)


# Eye base positions
_LE_CX, _RE_CX = 0.30, 0.70
_EYE_CY = 0.45
_EYE_RX = 0.115
_EYE_RY = 0.28

EMOTIONS: dict[str, EmotionState] = {
    "neutral": EmotionState(
        left_eye=EyeParams(_LE_CX, _EYE_CY, _EYE_RX, _EYE_RY),
        right_eye=EyeParams(_RE_CX, _EYE_CY, _EYE_RX, _EYE_RY),
    ),
    "happy": EmotionState(
        left_eye=EyeParams(_LE_CX, _EYE_CY + 0.02, _EYE_RX, _EYE_RY, squint_bottom=0.20),
        right_eye=EyeParams(_RE_CX, _EYE_CY + 0.02, _EYE_RX, _EYE_RY, squint_bottom=0.20),
        mouth=MouthParams(0.5, 0.85, 0.14, 0.06, visible=True),
    ),
    "sad": EmotionState(
        left_eye=EyeParams(_LE_CX, _EYE_CY + 0.03, _EYE_RX * 0.9, _EYE_RY * 0.75, squint_top=0.15),
        right_eye=EyeParams(_RE_CX, _EYE_CY + 0.03, _EYE_RX * 0.9, _EYE_RY * 0.75, squint_top=0.15),
        left_brow=BrowParams(_LE_CX, _EYE_CY - 0.30, 0.09, angle=15.0, visible=True),
        right_brow=BrowParams(_RE_CX, _EYE_CY - 0.30, 0.09, angle=-15.0, visible=True),
        mouth=MouthParams(0.5, 0.85, 0.10, -0.04, visible=True),
    ),
    "angry": EmotionState(
        left_eye=EyeParams(_LE_CX, _EYE_CY, _EYE_RX, _EYE_RY * 0.7, squint_top=0.25),
        right_eye=EyeParams(_RE_CX, _EYE_CY, _EYE_RX, _EYE_RY * 0.7, squint_top=0.25),
        left_brow=BrowParams(_LE_CX, _EYE_CY - 0.26, 0.10, angle=-20.0, visible=True),
        right_brow=BrowParams(_RE_CX, _EYE_CY - 0.26, 0.10, angle=20.0, visible=True),
        mouth=MouthParams(0.5, 0.85, 0.10, -0.02, visible=True),
    ),
    "surprised": EmotionState(
        left_eye=EyeParams(_LE_CX, _EYE_CY - 0.02, _EYE_RX * 1.15, _EYE_RY * 1.2),
        right_eye=EyeParams(_RE_CX, _EYE_CY - 0.02, _EYE_RX * 1.15, _EYE_RY * 1.2),
        mouth=MouthParams(0.5, 0.90, 0.08, 0.0, openness=0.08, visible=True),
    ),
    "sleeping": EmotionState(
        left_eye=EyeParams(_LE_CX, _EYE_CY + 0.05, _EYE_RX, _EYE_RY, openness=0.0),
        right_eye=EyeParams(_RE_CX, _EYE_CY + 0.05, _EYE_RX, _EYE_RY, openness=0.0),
        extra=ExtraAnimation(kind="sleeping_zzz"),
    ),
    "thinking": EmotionState(
        left_eye=EyeParams(_LE_CX + 0.03, _EYE_CY - 0.02, _EYE_RX * 0.9, _EYE_RY * 0.8, squint_bottom=0.15),
        right_eye=EyeParams(_RE_CX + 0.03, _EYE_CY - 0.04, _EYE_RX * 1.05, _EYE_RY * 0.9),
        left_brow=BrowParams(_LE_CX + 0.03, _EYE_CY - 0.28, 0.09, angle=5.0, visible=True),
        right_brow=BrowParams(_RE_CX + 0.03, _EYE_CY - 0.32, 0.09, angle=-8.0, visible=True),
        mouth=MouthParams(0.55, 0.85, 0.06, 0.0, visible=True),
        extra=ExtraAnimation(kind="thinking_dots"),
    ),
}

# ── Interpolation ───────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_bool(a: bool, b: bool, t: float) -> bool:
    return b if t > 0.5 else a


def _lerp_str(a: str, b: str, t: float) -> str:
    return b if t > 0.5 else a


def _lerp_eye(a: EyeParams, b: EyeParams, t: float) -> EyeParams:
    return EyeParams(
        _lerp(a.cx, b.cx, t), _lerp(a.cy, b.cy, t),
        _lerp(a.rx, b.rx, t), _lerp(a.ry, b.ry, t),
        _lerp(a.openness, b.openness, t),
        _lerp(a.squint_top, b.squint_top, t),
        _lerp(a.squint_bottom, b.squint_bottom, t),
    )


def _lerp_brow(a: BrowParams, b: BrowParams, t: float) -> BrowParams:
    return BrowParams(
        _lerp(a.cx, b.cx, t), _lerp(a.cy, b.cy, t),
        _lerp(a.width, b.width, t), _lerp(a.angle, b.angle, t),
        _lerp_bool(a.visible, b.visible, t),
    )


def _lerp_mouth(a: MouthParams, b: MouthParams, t: float) -> MouthParams:
    return MouthParams(
        _lerp(a.cx, b.cx, t), _lerp(a.cy, b.cy, t),
        _lerp(a.width, b.width, t), _lerp(a.height, b.height, t),
        _lerp(a.openness, b.openness, t),
        _lerp_bool(a.visible, b.visible, t),
    )


def lerp_emotion(a: EmotionState, b: EmotionState, t: float) -> EmotionState:
    t = max(0.0, min(1.0, t))
    return EmotionState(
        left_eye=_lerp_eye(a.left_eye, b.left_eye, t),
        right_eye=_lerp_eye(a.right_eye, b.right_eye, t),
        left_brow=_lerp_brow(a.left_brow, b.left_brow, t),
        right_brow=_lerp_brow(a.right_brow, b.right_brow, t),
        mouth=_lerp_mouth(a.mouth, b.mouth, t),
        extra=ExtraAnimation(kind=_lerp_str(a.extra.kind, b.extra.kind, t)),
    )


# ── Easing ──────────────────────────────────────────────────────────

def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


# ── Drawing ─────────────────────────────────────────────────────────

EYE_STROKE = 18  # pixels


def _draw_eye(surface: pygame.Surface, eye: EyeParams):
    cx = int(eye.cx * SCREEN_W)
    cy = int(eye.cy * SCREEN_H)
    rx = int(eye.rx * SCREEN_W)
    ry = int(eye.ry * SCREEN_H)

    if rx < 2:
        return

    # Fully closed: draw a curved arc line
    if eye.openness < 0.05 and ry >= 2:
        curve_depth = int(ry * 0.25)
        points = []
        for i in range(21):
            t = i / 20.0
            x = cx - rx + int(t * rx * 2)
            y = cy + int(curve_depth * math.sin(t * math.pi))
            points.append((x, y))
        pygame.draw.lines(surface, EYE_COLOR, False, points, EYE_STROKE // 3)
        return

    if ry < 2:
        return

    # Draw filled ellipse then cut out inner to make outline,
    # with optional top/bottom clipping for squint/openness
    size = (rx * 2, ry * 2)
    outer = pygame.Surface(size, pygame.SRCALPHA)
    inner = pygame.Surface(size, pygame.SRCALPHA)

    pygame.draw.ellipse(outer, EYE_COLOR, (0, 0, size[0], size[1]))
    inner_margin = EYE_STROKE
    pygame.draw.ellipse(inner, (0, 0, 0, 255),
                        (inner_margin, inner_margin,
                         size[0] - inner_margin * 2, size[1] - inner_margin * 2))
    outer.blit(inner, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

    # Eyelid clipping using elliptical arcs for natural look
    clip_top = eye.squint_top + (1.0 - eye.openness) * 0.5
    clip_bottom = eye.squint_bottom + (1.0 - eye.openness) * 0.5

    if clip_top > 0.01:
        lid = pygame.Surface(size, pygame.SRCALPHA)
        lid_ry = int(size[1] * (1.0 - clip_top))
        lid_rect = (0, -lid_ry, size[0], lid_ry * 2)
        pygame.draw.ellipse(lid, (0, 0, 0, 255), lid_rect)
        outer.blit(lid, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

    if clip_bottom > 0.01:
        lid = pygame.Surface(size, pygame.SRCALPHA)
        lid_ry = int(size[1] * (1.0 - clip_bottom))
        lid_rect = (0, size[1] - lid_ry, size[0], lid_ry * 2)
        pygame.draw.ellipse(lid, (0, 0, 0, 255), lid_rect)
        outer.blit(lid, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

    surface.blit(outer, (cx - rx, cy - ry))


def _draw_brow(surface: pygame.Surface, brow: BrowParams):
    if not brow.visible:
        return
    cx = int(brow.cx * SCREEN_W)
    cy = int(brow.cy * SCREEN_H)
    half_w = int(brow.width * SCREEN_W)
    angle_rad = math.radians(brow.angle)
    dy = int(math.sin(angle_rad) * half_w)

    start = (cx - half_w, cy + dy)
    end = (cx + half_w, cy - dy)
    pygame.draw.line(surface, EYE_COLOR, start, end, 6)


def _draw_mouth(surface: pygame.Surface, mouth: MouthParams):
    if not mouth.visible:
        return
    cx = int(mouth.cx * SCREEN_W)
    cy = int(mouth.cy * SCREEN_H)
    half_w = int(mouth.width * SCREEN_W)
    curve_h = int(mouth.height * SCREEN_H)
    open_h = int(mouth.openness * SCREEN_H)

    if open_h > 2:
        # Open mouth: ellipse
        rect = pygame.Rect(cx - half_w, cy - open_h, half_w * 2, open_h * 2)
        pygame.draw.ellipse(surface, EYE_COLOR, rect, 4)
    elif abs(curve_h) < 2:
        # Flat line
        pygame.draw.line(surface, EYE_COLOR, (cx - half_w, cy), (cx + half_w, cy), 4)
    else:
        # Curved line (smile or frown) via bezier approximation
        points = []
        for i in range(21):
            t = i / 20.0
            x = cx - half_w + t * half_w * 2
            y = cy + curve_h * 4 * t * (1 - t)
            points.append((int(x), int(y)))
        pygame.draw.lines(surface, EYE_COLOR, False, points, 4)


def _draw_thinking_dots(surface: pygame.Surface, t: float):
    """Three animated dots above the eyes."""
    y = int(0.08 * SCREEN_H)
    spacing = 40
    base_x = SCREEN_W // 2 - spacing

    for i in range(3):
        phase = (t * 2.0 + i * 0.4) % 1.0
        radius = int(4 + 6 * math.sin(phase * math.pi))
        alpha = int(120 + 135 * math.sin(phase * math.pi))
        x = base_x + i * spacing
        dot_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(dot_surf, (*EYE_COLOR, alpha), (radius, radius), radius)
        surface.blit(dot_surf, (x - radius, y - radius))


_zzz_font: pygame.font.Font | None = None


def _draw_sleeping_zzz(surface: pygame.Surface, t: float):
    """Floating Z's drifting up and to the right."""
    global _zzz_font
    if _zzz_font is None:
        _zzz_font = pygame.font.SysFont("sans", 1)

    base_x = int(0.75 * SCREEN_W)
    base_y = int(0.40 * SCREEN_H)

    for i in range(3):
        phase = (t * 0.3 + i * 0.33) % 1.0
        size = int(50 + 40 * phase)
        alpha = int(255 * (1.0 - phase * 0.7))
        x = base_x + int(phase * 120)
        y = base_y - int(phase * 250)

        font = pygame.font.SysFont("sans", size)
        z_surf = font.render("Z", True, EYE_COLOR)
        z_surf.set_alpha(alpha)
        surface.blit(z_surf, (x, y))


# ── Blink ───────────────────────────────────────────────────────────

BLINK_DURATION = 0.15  # seconds for full close+open
BLINK_INTERVAL_MIN = 2.0
BLINK_INTERVAL_MAX = 5.0


class BlinkController:
    def __init__(self):
        self._next_blink = time.monotonic() + 3.0
        self._blink_start: float | None = None

    def get_openness(self, now: float) -> float:
        if self._blink_start is not None:
            elapsed = now - self._blink_start
            if elapsed >= BLINK_DURATION:
                self._blink_start = None
                self._schedule_next(now)
                return 1.0
            half = BLINK_DURATION / 2
            if elapsed < half:
                return 1.0 - (elapsed / half)
            else:
                return (elapsed - half) / half
        if now >= self._next_blink:
            self._blink_start = now
            return 1.0
        return 1.0

    def _schedule_next(self, now: float):
        import random
        self._next_blink = now + random.uniform(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)


# ── Face renderer ───────────────────────────────────────────────────

TRANSITION_DURATION = 0.4  # seconds


class FaceRenderer:
    def __init__(self):
        self._current_emotion = "neutral"
        self._target_emotion = "neutral"
        self._current_state = EMOTIONS["neutral"]
        self._from_state = EMOTIONS["neutral"]
        self._to_state = EMOTIONS["neutral"]
        self._transition_start: float | None = None
        self._blink = BlinkController()

    @property
    def current_emotion(self) -> str:
        return self._target_emotion

    def set_emotion(self, name: str):
        if name not in EMOTIONS:
            return
        if name == self._target_emotion:
            return
        self._from_state = EmotionState(
            left_eye=EyeParams(self._current_state.left_eye.cx, self._current_state.left_eye.cy,
                               self._current_state.left_eye.rx, self._current_state.left_eye.ry,
                               self._current_state.left_eye.openness,
                               self._current_state.left_eye.squint_top,
                               self._current_state.left_eye.squint_bottom),
            right_eye=EyeParams(self._current_state.right_eye.cx, self._current_state.right_eye.cy,
                                self._current_state.right_eye.rx, self._current_state.right_eye.ry,
                                self._current_state.right_eye.openness,
                                self._current_state.right_eye.squint_top,
                                self._current_state.right_eye.squint_bottom),
            left_brow=BrowParams(self._current_state.left_brow.cx, self._current_state.left_brow.cy,
                                 self._current_state.left_brow.width, self._current_state.left_brow.angle,
                                 self._current_state.left_brow.visible),
            right_brow=BrowParams(self._current_state.right_brow.cx, self._current_state.right_brow.cy,
                                  self._current_state.right_brow.width, self._current_state.right_brow.angle,
                                  self._current_state.right_brow.visible),
            mouth=MouthParams(self._current_state.mouth.cx, self._current_state.mouth.cy,
                              self._current_state.mouth.width, self._current_state.mouth.height,
                              self._current_state.mouth.openness, self._current_state.mouth.visible),
            extra=ExtraAnimation(kind=self._current_state.extra.kind),
        )
        self._to_state = EMOTIONS[name]
        self._target_emotion = name
        self._transition_start = time.monotonic()

    def draw(self, surface: pygame.Surface):
        now = time.monotonic()

        # Advance transition
        if self._transition_start is not None:
            elapsed = now - self._transition_start
            t = min(1.0, elapsed / TRANSITION_DURATION)
            t = ease_in_out_cubic(t)
            self._current_state = lerp_emotion(self._from_state, self._to_state, t)
            if t >= 1.0:
                self._transition_start = None
                self._current_emotion = self._target_emotion

        state = self._current_state

        # Apply blink (skip when sleeping)
        blink_openness = 1.0 if self._target_emotion == "sleeping" else self._blink.get_openness(now)
        left = EyeParams(state.left_eye.cx, state.left_eye.cy,
                         state.left_eye.rx, state.left_eye.ry,
                         state.left_eye.openness * blink_openness,
                         state.left_eye.squint_top, state.left_eye.squint_bottom)
        right = EyeParams(state.right_eye.cx, state.right_eye.cy,
                          state.right_eye.rx, state.right_eye.ry,
                          state.right_eye.openness * blink_openness,
                          state.right_eye.squint_top, state.right_eye.squint_bottom)

        surface.fill(BG_COLOR)
        _draw_eye(surface, left)
        _draw_eye(surface, right)
        _draw_brow(surface, state.left_brow)
        _draw_brow(surface, state.right_brow)
        _draw_mouth(surface, state.mouth)

        if state.extra.kind == "thinking_dots":
            _draw_thinking_dots(surface, now % 10.0)
        elif state.extra.kind == "sleeping_zzz":
            _draw_sleeping_zzz(surface, now % 10.0)
