import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import os


class StringArtGenerator:
    def __init__(self, image_path, num_pins=288):
        self.num_pins = int(num_pins)
        self.img_size = 480
        self.original_image = Image.open(image_path)
        self.image_array = self._prepare_image()

        self.radius = (self.img_size - 1) / 2.0
        self.pins = self._calculate_pins()

        self.line_cache = {}
        self._precompute_all_lines()

    def _prepare_image(self):
        img = self.original_image.convert('L')
        min_dim = min(img.size)
        left = (img.size[0] - min_dim) / 2
        top = (img.size[1] - min_dim) / 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)

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
            # Upewniamy się, że piny nie wychodzą poza tablicę
            x = max(0, min(x, self.img_size - 1))
            y = max(0, min(y, self.img_size - 1))
            pins.append((x, y))
        return pins

    def _precompute_all_lines(self):
        """Generuje współrzędne pikseli dla każdej pary raz."""
        for i in range(self.num_pins):
            for j in range(i + 20, self.num_pins):
                x1, y1 = self.pins[i]
                x2, y2 = self.pins[j]
                length = int(math.hypot(x2 - x1, y2 - y1))
                if length == 0: continue

                y_coords = np.linspace(y1, y2, length).astype(int)
                x_coords = np.linspace(x1, x2, length).astype(int)
                self.line_cache[(i, j)] = (y_coords, x_coords)

    def generate(self, lines_to_draw=3000):
        # result_image musi być float32, żeby sumować setki jasnych linii
        result_image = np.full((self.img_size, self.img_size), 255, dtype=np.float32)

        current_pin = 0
        sequence = [current_pin]

        # PARAMETRY WYGLĄDU:
        penalty = 40  # Jak mocno "wybielamy" źródło po wybraniu linii
        line_shadow = 15  # Jak ciemna jest pojedyncza nitka (im mniej, tym subtelniejszy obraz)

        for _ in range(int(lines_to_draw)):
            best_pin = -1
            best_score = 256.0  # Szukamy najniższej średniej (najciemniejszej ścieżki)

            # Szukamy najlepszego kolejnego pinu
            for next_pin in range(self.num_pins):
                dist = min(abs(next_pin - current_pin), self.num_pins - abs(next_pin - current_pin))
                if dist < 20: continue

                idx = tuple(sorted((current_pin, next_pin)))
                if idx not in self.line_cache: continue
                y, x = self.line_cache[idx]

                # Szybkie obliczenie średniej jasności wzdłuż linii
                score = np.mean(self.image_array[y, x])

                if score < best_score:
                    best_score = score
                    best_pin = next_pin

            if best_pin == -1: break

            # Aktualizacja: nakładamy linię w cache i na wynik
            idx = tuple(sorted((current_pin, best_pin)))
            y, x = self.line_cache[idx]

            # "Wybielamy" obraz źródłowy, żeby nie rysować w kółko tej samej linii
            self.image_array[y, x] = np.clip(self.image_array[y, x] + penalty, 0, 255)

            # Rysujemy nitkę na czarnym obrazie wynikowym (odejmujemy od bieli)
            result_image[y, x] = np.clip(result_image[y, x] - line_shadow, 0, 255)

            sequence.append(best_pin)
            current_pin = best_pin

        # Zapisywanie
        if not os.path.exists('static'): os.makedirs('static')

        # Konwersja float32 -> uint8 przed zapisem PNG
        final_png = result_image.astype(np.uint8)
        cv2.imwrite('static/string_art_result.png', final_png)

        with open('static/instrukcja.txt', 'w') as f:
            f.write(" -> ".join(map(str, sequence)))

        return {"sequence": sequence, "pins": self.pins}
