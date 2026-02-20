import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import os

class StringArtGenerator:
    def __init__(self, image_path, num_pins=288):
        self.num_pins = int(num_pins)
        # 1. Zmniejszamy rozdzielczość obrazu roboczego (np. do 200px)
        # To drastycznie przyspiesza operacje NumPy bez widocznej straty jakości artystycznej.
        self.working_size = 200 
        self.original_image = Image.open(image_path)
        self.image_array = self._prepare_image()
        
        self.img_size = self.image_array.shape[0]
        self.radius = (self.img_size - 1) / 2.0
        self.pins = self._calculate_pins()
        
        # 2. Pre-kalkulacja wszystkich możliwych linii (Cache)
        # Dzięki temu pętla główna będzie tylko czytać z pamięci, a nie liczyć geometrię.
        self.line_cache = {}
        self._precompute_all_lines()

    def _prepare_image(self):
        # Kadrowanie do koła i zmiana rozmiaru na mniejszy dla szybkości
        img = self.original_image.convert('L')
        min_dim = min(img.size)
        left = (img.size[0] - min_dim) / 2
        top = (img.size[1] - min_dim) / 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        
        # ZMNIEJSZENIE: Obliczamy na mniejszej kopii, wynik i tak będzie dobry
        img = img.resize((self.working_size, self.working_size), Image.Resampling.LANCZOS)
        
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
            pins.append((x, y))
        return pins

    def _precompute_all_lines(self):
        """Generuje współrzędne pikseli dla każdej możliwej pary pinów raz."""
        for i in range(self.num_pins):
            for j in range(i + 20, self.num_pins): # Pomijamy zbyt bliskie piny
                x1, y1 = self.pins[i]
                x2, y2 = self.pins[j]
                length = int(math.hypot(x2 - x1, y2 - y1))
                if length == 0: continue
                
                y_coords = np.linspace(y1, y2, length).astype(int)
                x_coords = np.linspace(x1, x2, length).astype(int)
                
                # Zapisujemy jako tuple, żeby zajmowało mniej RAMu
                self.line_cache[(i, j)] = (y_coords, x_coords)

    def generate(self, lines_to_draw=3000):
        # Obraz wynikowy może być większy (np. 800px) dla ładniejszego PNG
        result_size = 800
        result_image = np.full((result_size, result_size), 255, dtype=np.uint8)
        
        # Skalowanie pinów dla obrazu wynikowego
        res_scale = (result_size - 1) / (self.img_size - 1)
        res_pins = [(int(p[0] * res_scale), int(p[1] * res_scale)) for p in self.pins]

        current_pin = 0
        sequence = [current_pin]
        penalty = 30 

        for _ in range(int(lines_to_draw)):
            best_pin = -1
            max_score = -1.0 # Szukamy największej różnicy (najciemniejszej linii)

            for next_pin in range(self.num_pins):
                dist = min(abs(next_pin - current_pin), self.num_pins - abs(next_pin - current_pin))
                if dist < 20: continue

                # Pobieramy współrzędne z cache
                idx = tuple(sorted((current_pin, next_pin)))
                if idx not in self.line_cache: continue
                y, x = self.line_cache[idx]

                # Obliczamy "ciemność" linii – im niższa wartość średnia, tym ciemniejsza linia
                # Używamy 255 - średnia, żeby szukać MAX
                score = 255.0 - np.mean(self.image_array[y, x])
                
                if score > max_score:
                    max_score = score
                    best_pin = next_pin

            if best_pin == -1: break

            # Aktualizacja obrazu źródłowego (nakładamy "karę", żeby nie wybierać tej samej linii)
            idx = tuple(sorted((current_pin, best_pin)))
            y, x = self.line_cache[idx]
            self.image_array[y, x] = np.clip(self.image_array[y, x] + penalty, 0, 255)
            
            # Rysowanie na wynikowym obrazie PNG (używamy OpenCV dla szybkości)
            cv2.line(result_image, res_pins[current_pin], res_pins[best_pin], 0, 1, cv2.LINE_AA)
            
            sequence.append(best_pin)
            current_pin = best_pin

        # Zapisywanie wyników
        if not os.path.exists('static'): os.makedirs('static')
        cv2.imwrite('static/string_art_result.png', result_image)
        
        with open('static/instrukcja.txt', 'w') as f:
            f.write(" -> ".join(map(str, sequence)))

        return {"sequence": sequence, "pins": [[int(p[0] * res_scale), int(p[1] * res_scale)] for p in self.pins]}
