from __future__ import annotations
from __utils import get_cv2_video_capture
import cv2
import numpy as np
import time

class Slider:
    def __init__(self, image_width, image_height, max_value, padding=20, slider_height=20, add_frame_count:bool=True):
        max_amount_of_numbers = int(np.ceil(np.log10(max_value)))
        self.text_padding = max_amount_of_numbers * 10
        self.image_height = image_height
        self.image_width = image_width
        self.padding = padding
        self.slider_x = self.padding*2 + self.text_padding
        self.slider_width = image_width - self.slider_x - self.padding
        self.slider_height = slider_height
        self.slider_y = image_height - self.padding - self.slider_height
        self.indicator_radius = 10 if (image_width < 750) else 15
        self.indicator_radius_original = self.indicator_radius
        self.min_value = 0
        self.max_value = max_value
        self.slider_value = 0

        self.mouse_over_slider = False

        self.indicator_x = np.inf
        self.indicator_y = np.inf
        self.buffer = self.indicator_radius
        self.is_dragging = False
        self.max_indicator_multiplier = 1.5
        self.add_frame_count = add_frame_count

        self.color_slider_complete = tuple(reversed((230, 200, 150)))  # tuple(reversed( --> cv2 BGR shenanigans
        self.color_slider =    tuple(reversed((220, 220, 220)))
        self.color_indicator = tuple(reversed((254, 151, 0)))
        self.color_indicator_shadow = tuple(reversed((184, 111, 4)))
        self.color_text =      tuple(reversed((220, 220, 220)))
        self.color_text_box = tuple(reversed((40, 40, 40)))

    def is_mouse_over_slider(self, mouse_x, mouse_y):
        if not (mouse_x >= self.slider_x - self.buffer):
            return False
        elif not (mouse_x <= self.slider_x + self.slider_width + self.buffer):
            return False
        elif not (mouse_y >= self.slider_y - self.buffer):
            return False
        elif not (mouse_y <= self.slider_y + self.slider_height + self.buffer):
            return False
        return True

    def is_mouse_over_indicator(self, mouse_x, mouse_y):
        x, y, r = self.indicator_x, self.indicator_y, self.indicator_radius
        distance = ((mouse_x - x) ** 2 + (mouse_y - y) ** 2) ** 0.5
        r_buffed = (r + self.buffer)
        if distance <= r_buffed:
            denominator = max(1e-6, (distance/r_buffed))
            size_multiplier = max(min(self.max_indicator_multiplier, 1.0/denominator), 1.0)
            return True, size_multiplier
        return False, 1.0

    def update_slider_value(self, mouse_x):
        slider_value = round((mouse_x - self.slider_x) / self.slider_width * self.max_value)
        self.slider_value = max(0, min(self.max_value, slider_value))

    def handle_mouse_input(self, event, mouse_x, mouse_y):
        left_pressed_down = event == cv2.EVENT_LBUTTONDOWN
        left_released = event == cv2.EVENT_LBUTTONUP
        mouse_moved = event == cv2.EVENT_MOUSEMOVE
        self.mouse_over_slider = self.is_mouse_over_slider(mouse_x, mouse_y)

        if left_pressed_down and self.mouse_over_slider:
            if not self.is_dragging:
                self.update_slider_value(mouse_x)
            self.is_dragging = True
        elif left_released:
            self.is_dragging = False
        elif mouse_moved and self.is_dragging:
            self.update_slider_value(mouse_x)


        if self.is_dragging:
            self.indicator_radius = int(self.indicator_radius_original * self.max_indicator_multiplier)
        elif not self.is_dragging:
            mouse_over_indicator, size_multiplier = self.is_mouse_over_indicator(mouse_x, mouse_y)
            self.indicator_radius = int(self.indicator_radius_original * size_multiplier)
        else:
            self.indicator_radius = self.indicator_radius_original

    def draw_frame_count(self, image):
        # Settings
        text = str(self.slider_value+1)
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1
        thickness = 1

        # Rectangle
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        text_x = 0
        text_y = text_height + 5
        cv2.rectangle(image, (text_x, 0), (text_x + text_width, text_y + 5), self.color_text_box, -1)

        # Draw the white text on top of the black box
        cv2.putText(image, text, (text_x, text_y), font, scale, self.color_text, thickness, cv2.LINE_AA)

    def draw_slider(self, image):
        # Draw frame count (upper left corner)
        if self.add_frame_count:
            self.draw_frame_count(image)

        # Draw frame count (next to play bar)
        text_p = (self.padding, self.slider_y + self.slider_height)
        cv2.putText(image, str(self.slider_value+1), text_p, 1, 1, self.color_text, 1, cv2.LINE_AA)

        # Draw slider bar (complete)
        rect_p1 = (self.slider_x, self.slider_y)
        rect_p2 = (self.slider_x + self.slider_width, self.slider_y + self.slider_height)
        cv2.rectangle(image, rect_p1, rect_p2, self.color_slider_complete, -1)

        # Draw slider bar
        self.indicator_x = int(self.slider_x + (self.slider_value / self.max_value) * self.slider_width)
        rect_p1 = (self.indicator_x, self.slider_y)
        rect_p2 = (self.slider_x + self.slider_width, self.slider_y + self.slider_height)
        cv2.rectangle(image, rect_p1, rect_p2, self.color_slider, -1)

        # Draw slider indicator
        self.indicator_y = self.slider_y + self.slider_height // 2
        cv2.circle(image, (self.indicator_x, self.indicator_y), self.indicator_radius+1, self.color_indicator_shadow, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (self.indicator_x, self.indicator_y), self.indicator_radius, self.color_indicator, -1, lineType=cv2.LINE_AA)


class VideoStream:
    def __init__(self, source:str | np.ndarray):
        self.is_cv2 = False
        self.video_path = None

        if isinstance(source, str):
            cap, info = get_cv2_video_capture(source)
            self.video_path = source
            self.is_cv2 = True
            self.source = cap
            self.info = info
            self.video_width =  info["video_width"]
            self.video_height = info["video_height"]
            self.total_frames = info["frame_count"]

        else:
            assert len(source.shape) == 4, f"Expected `source` to be of shape (frames, height, width, channels), but found {source.shape}"
            fn,h,w,c = source.shape
            assert c == 3, "Expected color images"
            self.source = source[:, :, :, ::-1]  # ::-1 is a fast RGB --> BGR
            self.video_width = w
            self.video_height = h
            self.total_frames = fn
        self.frame_index = 0

    def read(self):
        if self.is_cv2:
            successful, frame = self.source.read()
        else:
            frame = self.source[self.frame_index]
            successful = True
        self.frame_index += 1
        return successful, frame

    def release(self):
        if self.is_cv2:
            self.source.release()
        else:
            del self.source

    def set(self, flag, frame_index):
        if self.is_cv2:
            self.source.set(flag, frame_index)
            self.frame_index = self.source.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            self.frame_index = frame_index


class VideoPlayer:
    def __init__(self, video_source:str | np.ndarray, add_frame_count:bool=True):
        # cv2
        self.cap = VideoStream(video_source)
        self.window_name = "Video Player"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse, param=self)

        # video info
        self.video_path = self.cap.video_path
        self.video_width =  self.cap.video_width
        self.video_height = self.cap.video_height
        self.total_frames = self.cap.total_frames

        # Slider settings
        slider_height = 10
        padding = 20
        if (self.video_height > 800) or (self.video_width > 1000):
            padding = 30

        # General settings
        self.pad_size = (slider_height + padding * 2)
        self.color_pad_bottom = (100, 100, 100)
        self.min_video_width = 600
        self.side_pad = 0
        if self.video_width < self.min_video_width:
            self.side_pad = abs(self.video_width - self.min_video_width) // 2
        self.color_pad_sides = (20, 20, 20)

        # Mappings         escape       q                        a        left arrow    d         right arrow  1     2     3     4     5     6     7     8     9
        self.key_mapper = {113: "exit", 27: "exit", 32: "space", 97: "a", 2424832: "a", 100: "d", 2555904:"d", 49:1, 50:2, 51:3, 52:4, 53:5, 54:6, 55:7, 56:8, 57:9}
        self.fps_mapper =      {1: 1, 2: 3, 3: 5, 4: 15, 5: 30, 6: 60, 7: 120, 8:240, 9:999}
        self.fps_2_step_size = {v:k for k,v in self.fps_mapper.items()}

        # Adjustable variables
        self.current_frame = None
        self.current_frame_raw = None
        self.fps = self.fps_mapper[5]
        self.is_paused = False
        self.terminated = False

        # Adjust fps according to video length
        if self.total_frames <= 15:
            self.fps = self.fps_mapper[2]
        elif self.total_frames <= 30:
            self.fps = self.fps_mapper[3]
        elif self.total_frames <= 60:
            self.fps = self.fps_mapper[4]

        # Slider
        self.slider = Slider(
            self.video_width + self.side_pad * 2,
            self.video_height + self.pad_size,
            self.total_frames-1,
            padding,
            slider_height,
            add_frame_count
        )

        self.run()

    @staticmethod
    def on_mouse(event, x, y, _, params):
        self = params
        self.slider.handle_mouse_input(event, x, y)

    def pad_image(self, frame):
        if self.side_pad:
            frame = cv2.copyMakeBorder(frame, 0, 0, self.side_pad, self.side_pad, cv2.BORDER_CONSTANT, value=self.color_pad_sides)
        return cv2.copyMakeBorder(frame, 0, self.pad_size, 0, 0, cv2.BORDER_CONSTANT, value=self.color_pad_bottom)

    def next_frame(self, increment:bool):
        successful, frame = self.cap.read()
        assert successful, "This should always be successful, something unexpected has happend"
        self.current_frame_raw = frame.copy()
        self.current_frame = self.pad_image(frame)
        if increment and ((self.slider.slider_value-1) < self.total_frames):
            self.slider.slider_value += 1
        else:
            self.set_frame_index(self.slider.slider_value)

    def set_frame_index(self, frame_index:int):
        assert (frame_index >= 0) and (frame_index <= self.total_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.slider.slider_value = frame_index

    def clean_up(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.terminated = True

    def handle_key_input(self):
        # Extract and map keyboard inputs
        delay = int(1000 / self.fps) if not (self.is_paused or self.slider.is_dragging) else 5
        key_raw = cv2.waitKeyEx(delay)
        key = self.key_mapper.get(key_raw)
        close_button_pressed = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1

        # Handle input
        current_frame_index = self.slider.slider_value
        frame_step_size = self.fps_2_step_size[self.fps]
        slowest_fps = list(self.fps_mapper.values())[0]
        frame_step_size_extra = 1 if (self.fps == slowest_fps and (not self.is_paused)) else 0

        if close_button_pressed or (key == "exit"):
            self.clean_up()
        elif key == "space":
            self.is_paused = not self.is_paused
        elif key == "a":
            frame_index = max(0, current_frame_index - (frame_step_size + frame_step_size_extra))
            self.set_frame_index(frame_index)
        elif key == "d":
            frame_index = min(self.total_frames-1, self.slider.slider_value + (frame_step_size - frame_step_size_extra))
            self.set_frame_index(frame_index)
        elif key in range(1, 10):
            self.fps = self.fps_mapper[key]

        if key in ["a", "d"]:
            cv2.waitKeyEx(1) # ensures that "a" and "d" gets shown as fast a possible
            return True, key
        elif key_raw != -1:
            time.sleep(1.0 / self.fps)
        return False, key

    def _run(self):
        while not self.terminated:
            manual_frame_skip_detected, key  = self.handle_key_input()
            if self.terminated:
                break

            # Restart video if end is reached while not paused or dragging
            if (self.slider.slider_value >= (self.total_frames-1)) and (not self.slider.is_dragging) and (not self.is_paused):
                self.set_frame_index(0)
                self.next_frame(increment=False)
            elif manual_frame_skip_detected:
                do_increment = not self.is_paused
                if (key == "a") and (self.slider.slider_value == 0):
                    do_increment = False
                self.next_frame(increment=do_increment) # TODO: rewrite `increment`, the reason it works is too convoluted
            elif self.is_paused and (not self.slider.is_dragging):
                self.current_frame = self.pad_image(self.current_frame_raw)
            elif self.slider.is_dragging:
                if self.slider.slider_value != self.cap.frame_index:
                    frame_index = max(0, min(self.total_frames-1, self.slider.slider_value))
                    self.set_frame_index(frame_index)
                self.next_frame(increment=False)
            elif (not self.is_paused) and (not self.slider.is_dragging):
                self.next_frame(increment=True)

            self.slider.draw_slider(self.current_frame)
            # TODO add call back for frame processing
            cv2.imshow(self.window_name, self.current_frame)

            if self.cap.frame_index != self.slider.slider_value: # TODO: rewrite, they should not have separate "shared" state
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.slider.slider_value)

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.clean_up()
            raise RuntimeError(f"Fail to show video with error:\n{e}")