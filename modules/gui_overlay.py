"""Dear ImGui overlay for FoGaze — GLFW + OpenGL texture rendering."""

from __future__ import annotations

import cv2
import glfw
import numpy as np
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer


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

        imgui.create_context()
        self._impl = GlfwRenderer(self._window)

        self._width, self._height = win_w, win_h
        self._scene_tex = None
        self._scene_tex_w = self._scene_tex_h = 0
        self._face_tex = None

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

        # Face camera PIP (bottom-right, clear of left-panel instructions)
        if self._face_tex is not None:
            pw, ph = 320, 240
            px, py = self._width - pw - 10, self._height - ph - 10
            self._draw_texture_rect(self._face_tex, px, py, pw, ph)
            _draw_rect_outline(px, py, pw, ph, (1, 1, 1))

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
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)

    def update_face_texture(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        data = np.ascontiguousarray(rgb)

        if self._face_tex is not None:
            gl.glDeleteTextures([self._face_tex])
        self._face_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._face_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)

    # ── Cleanup ────────────────────────────────────────────────────────

    def close(self):
        for tex in [self._scene_tex, self._face_tex]:
            if tex is not None:
                gl.glDeleteTextures([tex])
        self._impl.shutdown()
        glfw.destroy_window(self._window)
        glfw.terminate()

    # ── Private helpers ────────────────────────────────────────────────

    def _draw_texture_full(self, tex_id: int, tex_w: int = 0, tex_h: int = 0):
        if tex_w == 0 or tex_h == 0:
            tex_w, tex_h = self._content_w, self._content_h
        tex_aspect = tex_w / tex_h
        win_aspect = self._width / self._height
        if tex_aspect > win_aspect:
            # Window is taller — letterbox top/bottom
            draw_w = self._width
            draw_h = self._width / tex_aspect
            ox = 0
            oy = (self._height - draw_h) / 2
        else:
            # Window is wider — letterbox left/right
            draw_h = self._height
            draw_w = self._height * tex_aspect
            ox = (self._width - draw_w) / 2
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


def _draw_rect_outline(x: int, y: int, w: int, h: int,
                       color: tuple[float, float, float]):
    gl.glDisable(gl.GL_TEXTURE_2D)
    gl.glColor4f(*color, 1)
    gl.glBegin(gl.GL_LINE_LOOP)
    gl.glVertex2f(x, y)
    gl.glVertex2f(x + w, y)
    gl.glVertex2f(x + w, y + h)
    gl.glVertex2f(x, y + h)
    gl.glEnd()
