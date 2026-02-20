import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import os


class StringArtGenerator:
    def __init__(self, image_path, num_pins=288, max_lines=5000):
        self.num_pins = num_pins
        self.max_lines = max_lines
        self.original_image = Image.open(image_path)
        self.image_array = self._prepare_image()
        self.img_size = self.image_array.shape[0]
        self.radius = (self.img_size - 1) / 2.0
        self.pins = self._calculate_pins()  # Oblicza listę (x, y) dla każdego pinu

    def _prepare_image(self):
        img = self.original_image.convert('L')
        min_dim = min(img.size)
        left = (img.size[0] - min_dim) / 2
        top = (img.size[1] - min_dim) / 2
        right = (img.size[0] + min_dim) / 2
        bottom = (img.size[1] + min_dim) / 2
        img = img.crop((left, top, right, bottom))
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, img.size[0], img.size[1]), fill=255)
        final_img = Image.new('L', img.size, 255)
        final_img.paste(img, mask=mask)
        return np.array(final_img, dtype=np.float32)

    def _calculate_pins(self):
        pins = []
        center = (self.img_size - 1) / 2.0
        for i in range(self.num_pins):
            angle = (2 * math.pi * i) / self.num_pins
            x = int(round(center + self.radius * math.cos(angle)))
            y = int(round(center + self.radius * math.sin(angle)))
            x = max(0, min(x, self.img_size - 1))
            y = max(0, min(y, self.img_size - 1))
            pins.append((x, y))
        return pins

    def _get_line_pixels(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        length = int(math.hypot(x2 - x1, y2 - y1))
        if length == 0: return [], []
        x_indices = np.linspace(x1, x2, length).astype(int)
        y_indices = np.linspace(y1, y2, length).astype(int)
        return y_indices, x_indices

    def generate(self, lines_to_draw=3000):
        result_image = np.full((self.img_size, self.img_size), 255, dtype=np.float32)
        current_pin = 0
        sequence = [current_pin]
        penalty = 20  # Wartość dodawana do obrazu źródłowego, by nie powtarzać tych samych linii
        line_shadow = 15

        for step in range(lines_to_draw):
            best_pin = -1
            best_score = 999.0
            for next_pin in range(self.num_pins):
                distance = min(abs(next_pin - current_pin), self.num_pins - abs(next_pin - current_pin))
                if distance < 20: continue
                y, x = self._get_line_pixels(self.pins[current_pin], self.pins[next_pin])
                if len(x) == 0: continue
                score = np.mean(self.image_array[y, x])  # Szukanie najciemniejszej ścieżki
                if score < best_score:
                    best_score = score
                    best_pin = next_pin

            if best_pin == -1: break
            sequence.append(best_pin)
            y, x = self._get_line_pixels(self.pins[current_pin], self.pins[best_pin])
            self.image_array[y, x] += penalty
            self.image_array[y, x] = np.clip(self.image_array[y, x], 0, 255)
            result_image[y, x] -= line_shadow
            result_image[y, x] = np.clip(result_image[y, x], 0, 255)
            current_pin = best_pin

        if not os.path.exists('static'): os.makedirs('static')
        cv2.imwrite('static/string_art_result.png', result_image.astype(np.uint8))
        with open('static/instrukcja.txt', 'w') as f:
            f.write(" -> ".join(map(str, sequence)))

        # ZWRACAMY DANE DLA ANIMACJI: piny i ich kolejność
        return {"sequence": sequence, "pins": self.pins}