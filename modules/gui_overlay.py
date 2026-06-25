"""Dear ImGui overlay for FoGaze — GLFW + OpenGL texture rendering."""

from __future__ import annotations

import cv2
import glfw
import numpy as np
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer


def apply_fogaze_theme():
    """Apply a clean modern dark theme matching FoGaze's cyan accent palette.

    Called once after the ImGui context is created.  Sets rounded corners,
    comfortable spacing, and a navy/cyan colour scheme so the panels look
    cohesive instead of the default ImGui grey.
    """
    style = imgui.get_style()

    # ── Geometry: rounded, roomy, flat ──────────────────────────────────
    style.window_rounding = 8.0
    style.child_rounding = 8.0
    style.frame_rounding = 6.0
    style.popup_rounding = 6.0
    style.grab_rounding = 6.0
    style.scrollbar_rounding = 8.0
    style.tab_rounding = 6.0
    style.window_border_size = 0.0
    style.frame_border_size = 0.0
    style.window_padding = (14, 14)
    style.frame_padding = (10, 7)
    style.item_spacing = (9, 9)
    style.item_inner_spacing = (7, 6)
    style.scrollbar_size = 12.0
    style.grab_min_size = 10.0
    style.window_title_align = (0.5, 0.5)

    # ── Colour palette (RGBA floats) ────────────────────────────────────
    CYAN     = (0.20, 0.78, 1.00, 1.00)
    CYAN_DIM = (0.20, 0.78, 1.00, 0.55)
    CYAN_LOW = (0.20, 0.78, 1.00, 0.28)
    BG_WIN   = (0.05, 0.06, 0.11, 0.96)
    BG_CHILD = (0.07, 0.08, 0.14, 1.00)
    BG_FRAME = (0.13, 0.15, 0.23, 1.00)
    BG_HOVER = (0.18, 0.21, 0.31, 1.00)
    BG_ACTIVE= (0.22, 0.26, 0.38, 1.00)
    TEXT     = (0.90, 0.91, 0.94, 1.00)
    TEXT_DIM = (0.50, 0.53, 0.60, 1.00)
    BORDER   = (0.16, 0.18, 0.26, 1.00)

    c = style.colors
    c[imgui.COLOR_TEXT]                    = TEXT
    c[imgui.COLOR_TEXT_DISABLED]           = TEXT_DIM
    c[imgui.COLOR_WINDOW_BACKGROUND]       = BG_WIN
    c[imgui.COLOR_CHILD_BACKGROUND]        = BG_CHILD
    c[imgui.COLOR_POPUP_BACKGROUND]        = BG_WIN
    c[imgui.COLOR_BORDER]                  = BORDER
    c[imgui.COLOR_FRAME_BACKGROUND]        = BG_FRAME
    c[imgui.COLOR_FRAME_BACKGROUND_HOVERED]= BG_HOVER
    c[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = BG_ACTIVE
    c[imgui.COLOR_TITLE_BACKGROUND]        = BG_CHILD
    c[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = BG_CHILD
    c[imgui.COLOR_BUTTON]                  = (0.16, 0.20, 0.30, 1.00)
    c[imgui.COLOR_BUTTON_HOVERED]          = (0.20, 0.55, 0.72, 1.00)
    c[imgui.COLOR_BUTTON_ACTIVE]           = CYAN
    c[imgui.COLOR_HEADER]                  = CYAN_LOW
    c[imgui.COLOR_HEADER_HOVERED]          = CYAN_DIM
    c[imgui.COLOR_HEADER_ACTIVE]           = CYAN_DIM
    c[imgui.COLOR_CHECK_MARK]              = CYAN
    c[imgui.COLOR_SLIDER_GRAB]             = CYAN
    c[imgui.COLOR_SLIDER_GRAB_ACTIVE]      = (0.45, 0.88, 1.00, 1.00)
    c[imgui.COLOR_SEPARATOR]               = BORDER
    c[imgui.COLOR_SCROLLBAR_BACKGROUND]    = (0.05, 0.06, 0.11, 0.0)
    c[imgui.COLOR_SCROLLBAR_GRAB]          = BG_HOVER
    c[imgui.COLOR_SCROLLBAR_GRAB_HOVERED]  = BG_ACTIVE
    c[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE]   = CYAN_DIM


class GUIOverlay:
    """Manages a fullscreen GLFW window with Dear ImGui overlay.

    Usage:
        gui = GUIOverlay(sw, sh)
        while not gui.should_close():
            gui.begin_frame()
            ... draw on canvas using cv2 ...
            gui.update_scene_texture(canvas)
            gui.update_face_texture(face_bgr)
            ... imgui widgets ...
            gui.render()
            if gui.was_key_pressed(glfw.KEY_D): ...
    """

    def __init__(self, width: int, height: int, title: str = "FoGaze",
                 fullscreen: bool = False):
        glfw.init()

        if fullscreen:
            glfw.window_hint(glfw.FLOATING, glfw.FALSE)
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
            glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            win_w, win_h = mode.size.width, mode.size.height
            self._window = glfw.create_window(win_w, win_h, title,
                                              monitor, None)
        else:
            glfw.window_hint(glfw.DECORATED, glfw.TRUE)
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
            mon = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(mon)
            max_w = int(mode.size.width * 0.85)
            max_h = int(mode.size.height * 0.85)
            win_w = min(width, max_w)
            win_h = int(win_w * height / width)
            if win_h > max_h:
                win_h = max_h
                win_w = int(win_h * width / height)
            self._window = glfw.create_window(win_w, win_h, title,
                                              None, None)
            # Center on screen (may fail on Wayland — non-fatal)
            try:
                mon = glfw.get_primary_monitor()
                mode = glfw.get_video_mode(mon)
                glfw.set_window_pos(
                    self._window,
                    (mode.size.width - win_w) // 2,
                    (mode.size.height - win_h) // 2,
                )
            except Exception:
                pass

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        self._content_w = width
        self._content_h = height
        # Left margin reserved for the side panel — the scene image is
        # rendered to the right of it so the panel never covers the view.
        self._margin_left = 0

        imgui.create_context()
        apply_fogaze_theme()
        self._impl = GlfwRenderer(self._window)

        self._width, self._height = win_w, win_h
        self._scene_tex = None
        self._scene_tex_w = self._scene_tex_h = 0
        self._face_tex = None
        self._depth_tex = None
        self._eye_tex = None

        # ── Key event handling (chain before GlfwRenderer) ──────────
        self._keys_pressed: set[int] = set()
        # Replace GlfwRenderer's callbacks to insert our own in front
        self._prev_key = glfw.set_key_callback(self._window, self._on_key)
        self._prev_char = glfw.set_char_callback(self._window, self._on_char)
        self._prev_mouse_btn = glfw.set_mouse_button_callback(
            self._window, self._on_mouse_button)
        self._prev_scroll = glfw.set_scroll_callback(
            self._window, self._on_scroll)
        self._prev_cursor = glfw.set_cursor_pos_callback(
            self._window, self._on_cursor)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def should_close(self) -> bool:
        return glfw.window_should_close(self._window)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def window(self):
        return self._window

    @property
    def margin_left(self) -> int:
        return self._margin_left

    @margin_left.setter
    def margin_left(self, value: int):
        self._margin_left = max(0, int(value))

    @property
    def face_texture_id(self):
        """OpenGL texture ID for the face camera, or None."""
        return self._face_tex

    @property
    def depth_texture_id(self):
        """OpenGL texture ID for the depth colormap, or None."""
        return self._depth_tex

    @property
    def eye_texture_id(self):
        """OpenGL texture ID for the eye tracking display, or None."""
        return self._eye_tex

    # ── Key / event API ────────────────────────────────────────────────

    def was_key_pressed(self, glfw_key: int) -> bool:
        """Check and consume a one-shot key press event."""
        if glfw_key in self._keys_pressed:
            self._keys_pressed.discard(glfw_key)
            return True
        return False

    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self._keys_pressed.add(key)
        elif action == glfw.RELEASE:
            self._keys_pressed.discard(key)
        # Forward to GlfwRenderer
        if self._prev_key:
            self._prev_key(window, key, scancode, action, mods)

    def _on_char(self, window, char):
        if self._prev_char:
            self._prev_char(window, char)

    def _on_mouse_button(self, window, button, action, mods):
        if self._prev_mouse_btn:
            self._prev_mouse_btn(window, button, action, mods)

    def _on_scroll(self, window, x_offset, y_offset):
        if self._prev_scroll:
            self._prev_scroll(window, x_offset, y_offset)

    def _on_cursor(self, window, xpos, ypos):
        if self._prev_cursor:
            self._prev_cursor(window, xpos, ypos)

    # ── Frame lifecycle ────────────────────────────────────────────────

    def begin_frame(self):
        glfw.poll_events()
        self._impl.process_inputs()
        imgui.new_frame()

    def render(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self._width, self._height, 0, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        if self._scene_tex is not None:
            self._draw_texture_full(self._scene_tex,
                                    self._scene_tex_w, self._scene_tex_h)

        # Face camera is rendered via imgui.image() inside the MainMenu panel.
        # No OpenGL-based face PIP here.

        # ImGui overlay
        imgui.render()
        self._impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self._window)

    # ── Texture uploads ────────────────────────────────────────────────

    def update_scene_texture(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        self._scene_tex_w, self._scene_tex_h = w, h
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        data = np.ascontiguousarray(rgb)

        if self._scene_tex is None:
            self._scene_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._scene_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        # Rows are tightly packed (no 4-byte padding); without this an
        # arbitrary-width RGB image (e.g. the eye crop) skews diagonally.
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)

    def update_face_texture(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        data = np.ascontiguousarray(rgb)

        if self._face_tex is None:
            self._face_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._face_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        # Rows are tightly packed (no 4-byte padding); without this an
        # arbitrary-width RGB image (e.g. the eye crop) skews diagonally.
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)

    def update_depth_texture(self, bgr: np.ndarray):
        """Upload a depth colormap BGR image as an OpenGL texture."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        data = np.ascontiguousarray(rgb)

        if self._depth_tex is None:
            self._depth_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._depth_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        # Rows are tightly packed (no 4-byte padding); without this an
        # arbitrary-width RGB image (e.g. the eye crop) skews diagonally.
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)

    def update_eye_texture(self, bgr: np.ndarray):
        """Upload an eye tracking BGR image as an OpenGL texture."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        data = np.ascontiguousarray(rgb)

        if self._eye_tex is None:
            self._eye_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._eye_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        # Rows are tightly packed (no 4-byte padding); without this an
        # arbitrary-width RGB image (e.g. the eye crop) skews diagonally.
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)

    # ── Cleanup ────────────────────────────────────────────────────────

    def close(self):
        for tex in [self._scene_tex, self._face_tex, self._depth_tex, self._eye_tex]:
            if tex is not None:
                gl.glDeleteTextures([tex])
        self._impl.shutdown()
        glfw.destroy_window(self._window)
        glfw.terminate()

    # ── Private helpers ────────────────────────────────────────────────

    def _draw_texture_full(self, tex_id: int, tex_w: int = 0, tex_h: int = 0):
        if tex_w == 0 or tex_h == 0:
            tex_w, tex_h = self._content_w, self._content_h

        # Available region = whole window minus the reserved left panel.
        region_x = self._margin_left
        region_w = max(1, self._width - self._margin_left)
        region_h = self._height

        tex_aspect = tex_w / tex_h
        win_aspect = region_w / region_h
        if tex_aspect > win_aspect:
            # Region is taller — letterbox top/bottom
            draw_w = region_w
            draw_h = region_w / tex_aspect
            ox = region_x
            oy = (region_h - draw_h) / 2
        else:
            # Region is wider — letterbox left/right
            draw_h = region_h
            draw_w = region_h * tex_aspect
            ox = region_x + (region_w - draw_w) / 2
            oy = 0
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glColor4f(1, 1, 1, 1)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0)
        gl.glVertex2f(ox, oy)
        gl.glTexCoord2f(1, 0)
        gl.glVertex2f(ox + draw_w, oy)
        gl.glTexCoord2f(1, 1)
        gl.glVertex2f(ox + draw_w, oy + draw_h)
        gl.glTexCoord2f(0, 1)
        gl.glVertex2f(ox, oy + draw_h)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

    @staticmethod
    def _draw_texture_rect(tex_id: int, x: int, y: int, w: int, h: int):
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glColor4f(1, 1, 1, 1)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0)
        gl.glVertex2f(x, y)
        gl.glTexCoord2f(1, 0)
        gl.glVertex2f(x + w, y)
        gl.glTexCoord2f(1, 1)
        gl.glVertex2f(x + w, y + h)
        gl.glTexCoord2f(0, 1)
        gl.glVertex2f(x, y + h)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)
