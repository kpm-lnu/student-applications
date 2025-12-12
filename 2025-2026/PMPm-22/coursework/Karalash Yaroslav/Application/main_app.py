import pygame
import sys
import json
import numpy as np
import random
import math 
import simulation_engine  
import matplotlib.pyplot as plt 

SIM_PANEL_WIDTH = 300 
MAP_WIDTH = 800
MAP_HEIGHT = 600
SCREEN_WIDTH = MAP_WIDTH + SIM_PANEL_WIDTH
SCREEN_HEIGHT = MAP_HEIGHT
MAP_PADDING = 20

COLOR_BACKGROUND = (240, 240, 240) 
COLOR_UI_PANEL = (225, 225, 225)   
COLOR_MAP_FILL = (255, 255, 255)     
COLOR_MAP_BORDER = (50, 50, 50)       
COLOR_MAP_HOVER = (220, 230, 255)     
COLOR_MAP_SELECTED = (180, 255, 200)  
COLOR_GRAPH_BUTTON = (0, 150, 0)
COLOR_HANTA_DOT = (0, 0, 255, 200)    
COLOR_LEPTO_DOT = (255, 0, 0, 200)    
COLOR_TEXT = (10, 10, 10)
COLOR_INPUT_BOX = (255, 255, 255)
COLOR_INPUT_BOX_ACTIVE = (240, 255, 240)
COLOR_INPUT_BORDER = (100, 100, 100)
COLOR_TOOLTIP_BG = (255, 255, 220) 

DOT_TO_PEOPLE_RATIO = 1000.0
DOT_RADIUS = 2 

NAME_MAPPER = {
    "Автономна республіка Крим": "Автономна Республіка Крим", "Севастополь": "м. Севастополь",
    "Київ": "м. Київ", "Київська": "Київська область",
    "Вінницька": "Вінницька область", "Волинська": "Волинська область",
    "Дніпропетровська": "Дніпропетровська область", "Донецька": "Донецька область",
    "Житомирська": "Житомирська область", "Закарпатська": "Закарпатська область",
    "Запорізька": "Запорізька область", "Івано-Франківська": "Івано-Франківська область",
    "Кіровоградська": "Кіровоградська область", "Луганська": "Луганська область",
    "Львівська": "Львівська область", "Миколаївська": "Миколаївська область",
    "Одеська": "Одеська область", "Полтавська": "Полтавська область",
    "Рівненська": "Рівненська область", "Сумська": "Сумська область",
    "Тернопільська": "Тернопільська область", "Харківська": "Харківська область",
    "Херсонська": "Херсонська область", "Хмельницька": "Хмельницька область",
    "Черкаська": "Черкаська область", "Чернівецька": "Чернівецька область",
    "Чернігівська": "Чернігівська область"
}

def is_point_in_polygon(point, polygon):
    x, y = point
    num_vertices = len(polygon)
    is_inside = False
    p1_x, p1_y = polygon[0]
    for i in range(1, num_vertices + 1):
        p2_x, p2_y = polygon[i % num_vertices]
        if y > min(p1_y, p2_y):
            if y <= max(p1_y, p2_y):
                if x <= max(p1_x, p2_x):
                    if p1_y != p2_y:
                        x_intersection = (y - p1_y) * (p2_x - p1_x) / (p2_y - p1_y) + p1_x
                    if p1_x == p2_x or x <= x_intersection:
                        is_inside = not is_inside
        p1_x, p1_y = p2_x, p2_y
    return is_inside

class InputBox:
    def __init__(self, x, y, w, h, text='', font=None, is_float=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = COLOR_INPUT_BOX
        self.text = text
        self.font = font or pygame.font.SysFont('Arial', 18)
        self.txt_surface = self.font.render(text, True, COLOR_TEXT)
        self.active = False
        self.is_float = is_float 

    def set_active(self, is_active):
        self.active = is_active
        self.color = COLOR_INPUT_BOX_ACTIVE if self.active else COLOR_INPUT_BOX

    def handle_key_event(self, event):
        if not self.active:
            return

        if event.key == pygame.K_RETURN:
            self.set_active(False)
        elif event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        else:
            char = event.unicode
            if char.isdigit():
                self.text += char
            elif self.is_float and (event.key == pygame.K_PERIOD or event.key == pygame.K_KP_PERIOD or event.key == pygame.K_COMMA) and '.' not in self.text:
                self.text += '.' 
                
        self.txt_surface = self.font.render(self.text, True, COLOR_TEXT)

    def get_value(self, default=0):
        if self.is_float:
            try: return float(self.text)
            except ValueError: return float(default)
        else:
            try: return int(self.text)
            except ValueError: return int(default)

    def set_text(self, text):
        self.text = str(text)
        self.txt_surface = self.font.render(self.text, True, COLOR_TEXT)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, COLOR_INPUT_BORDER, self.rect, 1)
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))

class Oblast:
    def __init__(self, name, geo_polygons):
        self.name = name
        self.geo_polygons = geo_polygons 
        self.scaled_polygons = []        
        self.bounding_box = None         
        self.center = (0, 0)             
        self.initial_ir = 0.0
        self.initial_ia = 0.0
        self.initial_w = 0.0
        self.dot_cache = [] 
        
    def scale(self, scale_func):
        all_points = []
        for poly_part in self.geo_polygons:
            scaled_part = [scale_func(lon, lat) for lon, lat in poly_part]
            self.scaled_polygons.append(scaled_part)
            all_points.extend(scaled_part)
        
        if all_points:
            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)
            self.bounding_box = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            self.center = self.bounding_box.center
        
    def is_hovered(self, mouse_pos):
        if self.bounding_box and self.bounding_box.collidepoint(mouse_pos):
            for poly in self.scaled_polygons:
                if is_point_in_polygon(mouse_pos, poly):
                    return True
        return False
        
    def get_random_point_inside(self):
        if not self.dot_cache: 
            if not self.bounding_box: return (0, 0)
            return self.bounding_box.center
        return random.choice(self.dot_cache)

    def draw(self, surface, color):
        for poly in self.scaled_polygons:
            if len(poly) > 2:
                pygame.draw.polygon(surface, color, poly)
                pygame.draw.polygon(surface, COLOR_MAP_BORDER, poly, 2)

class App:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Епідеміологічна Модель")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Arial', 12)
        self.font_medium = pygame.font.SysFont('Arial', 16)
        self.font_large = pygame.font.SysFont('Arial', 20, bold=True)
        
        self.map_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
        self.ui_surface = pygame.Surface((SIM_PANEL_WIDTH, SCREEN_HEIGHT))
        self.dots_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)

        self.app_state = "LOADING" 
        self.sim_data = None
        self.initial_conditions = None
        self.sim_results = None
        self.t_eval = None
        
        self.oblasts = {} 
        self.hovered_oblast_name = None
        self.selected_oblast_name = None 
        
        self.input_boxes = self._create_input_boxes()
        self.active_input_box = None
 
        self.current_day = 0
        self.is_paused = True
        self.base_days_per_second = 24.0 
        self.playback_speed = 1.0       
        self.day_accumulator = 0.0      
        self.needs_dot_update = True    
        self.simulation_finished = False 
        
        self.graph_selected_oblast = None
        self.graph_button_rect = None
        
        self.dot_lists = {}
        
        self.map_scale = 1.0
        self.map_offset_x = 0
        self.map_offset_y = 0
        self.lon_min = 180
        self.lon_max = -180
        self.lat_min = 90
        self.lat_max = -90
        self.lon_correction = 1.0 

    def _setup_scaling(self):
        canvas_width = MAP_WIDTH - 2 * MAP_PADDING
        canvas_height = MAP_HEIGHT - 2 * MAP_PADDING
        
        if (self.lon_max - self.lon_min) == 0 or (self.lat_max - self.lat_min) == 0:
            return

        center_lat_rad = math.radians((self.lat_min + self.lat_max) / 2)
        self.lon_correction = math.cos(center_lat_rad)
        
        geo_width = (self.lon_max - self.lon_min) * self.lon_correction
        geo_height = (self.lat_max - self.lat_min)

        if geo_width == 0 or geo_height == 0: return 
        
        scale_x = canvas_width / geo_width
        scale_y = canvas_height / geo_height
        
        self.map_scale = min(scale_x, scale_y)
    
        map_pixel_width = geo_width * self.map_scale
        map_pixel_height = geo_height * self.map_scale
        
        self.map_offset_x = (MAP_WIDTH - map_pixel_width) / 2
        self.map_offset_y = (MAP_HEIGHT - map_pixel_height) / 2
        
    def _scale_point(self, lon, lat):
        x = (lon - self.lon_min) * self.lon_correction * self.map_scale
        y = (self.lat_max - lat) * self.map_scale 
        return int(x + self.map_offset_x), int(y + self.map_offset_y)

    def _create_input_boxes(self):
        boxes = {}
        font = pygame.font.SysFont('Arial', 18)
        px, py, pw, ph = 20, 50, SIM_PANEL_WIDTH - 40, 28
        
        boxes['duration'] = InputBox(px, py + 30, pw, ph, '365', font, is_float=False)
        boxes['start_day'] = InputBox(px, py + 90, pw, ph, '1', font, is_float=False)
        
        py_oblast = py + 200
        boxes['oblast_ir'] = InputBox(px, py_oblast + 30, pw, ph, '0.0', font, is_float=True)
        boxes['oblast_ia'] = InputBox(px, py_oblast + 90, pw, ph, '0.0', font, is_float=True)
        boxes['oblast_w'] = InputBox(px, py_oblast + 150, pw, ph, '0.0', font, is_float=True)
        
        return boxes

    def load_data_and_map(self):
        
        self.sim_data = simulation_engine.load_simulation_data()
        if self.sim_data is None: return False
            
        self.initial_conditions = simulation_engine.create_initial_conditions(
            self.sim_data['n_regions']
        )
        
        try:
            with open('ua.json', 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
        except Exception as e:
            print(f"ПОМИЛКА: Не вдалося завантажити ua.json: {e}")
            return False

        all_features_data = {}
        
        for feature in geojson_data.get('features', []):
            oblast_name_json = feature.get('properties', {}).get('name')
            if not oblast_name_json: continue
            oblast_name_mapped = NAME_MAPPER.get(oblast_name_json)
            if not oblast_name_mapped: continue 
                
            geometry = feature.get('geometry', {})
            coordinates = geometry.get('coordinates', [])
            polygons_data = []
            if geometry.get('type') == 'Polygon':
                polygons_data = [coordinates]
            elif geometry.get('type') == 'MultiPolygon':
                polygons_data = coordinates

            oblast_geo_parts = []
            for poly in polygons_data:
                poly_part = poly[0] # [lon, lat]
                oblast_geo_parts.append(poly_part)
                for lon, lat in poly_part:
                    if lon < self.lon_min: self.lon_min = lon
                    if lon > self.lon_max: self.lon_max = lon
                    if lat < self.lat_min: self.lat_min = lat
                    if lat > self.lat_max: self.lat_max = lat
            
            all_features_data[oblast_name_mapped] = oblast_geo_parts
 
        self._setup_scaling() 
        
        print("Масштабування завершено. Генерую кеш цяток (це може зайняти час)...")
        
        sorted_oblast_names = sorted(all_features_data.keys(), key=lambda x: "м." in x)

        for name in sorted_oblast_names:
            geo_polys = all_features_data[name]
            oblast_obj = Oblast(name, geo_polys)
            oblast_obj.scale(self._scale_point) 
            self.oblasts[name] = oblast_obj
            self.dot_lists[name] = {"hanta": [], "lepto": []}
            
            if oblast_obj.bounding_box:
                points_generated = 0
                attempts = 0
                while points_generated < 1000 and attempts < 20000:
                    x = random.randint(oblast_obj.bounding_box.left, oblast_obj.bounding_box.right)
                    y = random.randint(oblast_obj.bounding_box.top, oblast_obj.bounding_box.bottom)
                    point = (x, y)
                    is_inside = False
                    for poly in oblast_obj.scaled_polygons:
                        if is_point_in_polygon(point, poly):
                            is_inside = True
                            break
                    if is_inside:
                        oblast_obj.dot_cache.append(point)
                        points_generated += 1
                    attempts += 1
        print("Карта та дані симуляції успішно завантажені.")
        return True


    def start_simulation(self):
        self.app_state = "SIMULATING"
        self.draw() 
        pygame.display.flip()
        
        sim_days = self.input_boxes['duration'].get_value(365)
        start_day = self.input_boxes['start_day'].get_value(1)
        
        print(f"[App] Підготовка початкових умов на {sim_days} днів...")
        
        for oblast_name, oblast_obj in self.oblasts.items():
            if oblast_obj.initial_ir > 0 or oblast_obj.initial_ia > 0 or oblast_obj.initial_w > 0:
                idx = self.sim_data["oblast_to_index"][oblast_name]
                self.initial_conditions[idx, simulation_engine.i_I_R] = oblast_obj.initial_ir
                self.initial_conditions[idx, simulation_engine.i_I_A] = oblast_obj.initial_ia
                self.initial_conditions[idx, simulation_engine.i_W] = oblast_obj.initial_w
                print(f"  > Додано в {oblast_name}: IR={oblast_obj.initial_ir}, IA={oblast_obj.initial_ia}, W={oblast_obj.initial_w}")

        self.sim_results, self.t_eval = simulation_engine.run_simulation(
            start_day,
            sim_days,
            self.initial_conditions,
            self.sim_data,
            simulation_engine.DEFAULT_PARAMS 
        )
        
        self.clock.tick()

        if self.sim_results is not None:
            print("[App] Симуляція завершена. Початок анімації.")
            self.app_state = "ANIMATING"
            self.is_paused = False 
            self.simulation_finished = False 
            self.playback_speed = 1.0 
            self.day_accumulator = 0.0
            self.current_day = max(0, start_day - 1)
            self.needs_dot_update = True 
        else:
            print("[App] ПОМИЛКА СИМУЛЯЦІЇ. Повернення до налаштувань.")
            self.app_state = "SETUP"

    def reset_simulation(self):
        print("[App] Скидання симуляції...")
        self.app_state = "SETUP"
        self.is_paused = True
        self.simulation_finished = False
        self.current_day = 0
        self.sim_results = None
        self.t_eval = None
        self.selected_oblast_name = None
        self.hovered_oblast_name = None
        self.graph_selected_oblast = None 
        self.graph_button_rect = None   
        
        self.initial_conditions = simulation_engine.create_initial_conditions(
            self.sim_data['n_regions']
        )
        for oblast in self.oblasts.values():
            oblast.initial_ir = 0.0
            oblast.initial_ia = 0.0
            oblast.initial_w = 0.0
        
        for name in self.dot_lists:
            self.dot_lists[name]["hanta"] = []
            self.dot_lists[name]["lepto"] = []
            
        self.input_boxes['oblast_ir'].set_text('0.0')
        self.input_boxes['oblast_ia'].set_text('0.0')
        self.input_boxes['oblast_w'].set_text('0.0')
        
        self.dots_surface.fill((0, 0, 0, 0))


    def handle_events(self):
        
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    
                    self.active_input_box = None 
                    clicked_on_ui = mouse_pos[0] >= MAP_WIDTH
                    
                    if clicked_on_ui:
                        ui_mouse_pos = (mouse_pos[0] - MAP_WIDTH, mouse_pos[1])
                        
                        if self.simulation_finished and self.graph_button_rect and self.graph_button_rect.collidepoint(ui_mouse_pos):
                            self.show_graph()
                        else:
                            for box in self.input_boxes.values():
                                if box.rect.collidepoint(ui_mouse_pos):
                                    box.set_active(True)
                                    self.active_input_box = box
                                else:
                                    box.set_active(False)
                    else:
                        for box in self.input_boxes.values():
                            box.set_active(False) 
                        
                        if self.app_state == "SETUP": 
                            if self.hovered_oblast_name:
                                self.selected_oblast_name = self.hovered_oblast_name
                                oblast = self.oblasts[self.selected_oblast_name]
                                self.input_boxes['oblast_ir'].set_text(oblast.initial_ir)
                                self.input_boxes['oblast_ia'].set_text(oblast.initial_ia)
                                self.input_boxes['oblast_w'].set_text(oblast.initial_w)
                        
                        elif self.app_state == "ANIMATING" and self.simulation_finished:
                            if self.hovered_oblast_name:
                                self.graph_selected_oblast = self.hovered_oblast_name
                            else:
                                self.graph_selected_oblast = None

            if event.type == pygame.KEYDOWN:
                if self.active_input_box:
                    self.active_input_box.handle_key_event(event)
                else:
                    if self.app_state == "ANIMATING":
                        if event.key == pygame.K_SPACE:
                            self.is_paused = not self.is_paused
                        
                        elif event.key == pygame.K_RIGHT or event.key == pygame.K_PERIOD: # > або .
                            self.playback_speed = min(self.playback_speed * 2.0, 64.0)
                        elif event.key == pygame.K_LEFT or event.key == pygame.K_COMMA: # < або ,
                            self.playback_speed = max(self.playback_speed / 2.0, 0.25)

                        elif event.key == pygame.K_r:
                            if self.is_paused or self.simulation_finished:
                                self.reset_simulation()
                    
                    elif self.app_state == "SETUP":
                        if event.key == pygame.K_RETURN:
                            self.start_simulation()

    def update(self, delta_time_ms):
        
        self.hovered_oblast_name = None
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos[0] < MAP_WIDTH: 
            for name, oblast in reversed(self.oblasts.items()):
                if oblast.is_hovered(mouse_pos):
                    self.hovered_oblast_name = name
                    break 
        
        self.needs_dot_update = False 
        
        if self.app_state == "ANIMATING" and not self.is_paused:
            
            delta_seconds = delta_time_ms / 1000.0
            days_to_advance = delta_seconds * self.base_days_per_second * self.playback_speed
            self.day_accumulator += days_to_advance
            
            days_to_increment = int(self.day_accumulator)
            
            if days_to_increment > 0 and not self.simulation_finished:
                if self.t_eval is None:
                    print("Помилка: t_eval is None, не можу продовжити анімацію.")
                    self.is_paused = True
                    return

                new_day = self.current_day + days_to_increment
                
                if new_day >= len(self.t_eval) - 1:
                    self.current_day = len(self.t_eval) - 1
                    self.is_paused = True
                    self.simulation_finished = True
                else:
                    self.current_day = new_day
                
                self.day_accumulator -= days_to_increment
                self.needs_dot_update = True 
        
        if self.needs_dot_update and self.app_state == "ANIMATING":
            self.update_dots()
            
        if self.app_state == "SETUP" and self.selected_oblast_name:
            oblast = self.oblasts[self.selected_oblast_name]
            oblast.initial_ir = self.input_boxes['oblast_ir'].get_value(0.0)
            oblast.initial_ia = self.input_boxes['oblast_ia'].get_value(0.0)
            oblast.initial_w = self.input_boxes['oblast_w'].get_value(0.0)

    def update_dots(self):
        if self.sim_results is None:
            return

        for oblast_name, oblast_obj in self.oblasts.items():
            idx = self.sim_data['oblast_to_index'][oblast_name]
            
            population = self.sim_data['population_data'].get(oblast_name, 1_000_000) 
            
            day_idx = min(int(self.current_day), self.sim_results.shape[2] - 1)
            
            i_h = simulation_engine.i_I_H
            fraction_hanta = self.sim_results[idx, i_h, day_idx]
            total_infected_hanta = fraction_hanta * population
            target_hanta = int(total_infected_hanta / DOT_TO_PEOPLE_RATIO)
            
            list_hanta = self.dot_lists[oblast_name]["hanta"]
            while len(list_hanta) < target_hanta:
                list_hanta.append(oblast_obj.get_random_point_inside())
            while len(list_hanta) > target_hanta:
                list_hanta.pop()

            i_l = simulation_engine.i_I_L
            fraction_lepto = self.sim_results[idx, i_l, day_idx]
            total_infected_lepto = fraction_lepto * population
            target_lepto = int(total_infected_lepto / DOT_TO_PEOPLE_RATIO)
            
            list_lepto = self.dot_lists[oblast_name]["lepto"]
            while len(list_lepto) < target_lepto:
                list_lepto.append(oblast_obj.get_random_point_inside())
            while len(list_lepto) > target_lepto:
                list_lepto.pop()

    def draw_map_surface(self):
        self.map_surface.fill(COLOR_BACKGROUND)
        
        for name, oblast in self.oblasts.items():
            color = COLOR_MAP_FILL
            if name == self.graph_selected_oblast: 
                color = (255, 180, 220) 
            elif name == self.selected_oblast_name:
                color = COLOR_MAP_SELECTED
            elif name == self.hovered_oblast_name:
                color = COLOR_MAP_HOVER
            oblast.draw(self.map_surface, color)
        
        if self.app_state == "ANIMATING":
            if self.needs_dot_update:
                self.dots_surface.fill((0, 0, 0, 0)) # Очищуємо
                for oblast_name in self.oblasts.keys():
                    for pos in self.dot_lists[oblast_name]["hanta"]:
                        pygame.draw.circle(self.dots_surface, COLOR_HANTA_DOT, pos, DOT_RADIUS)
                    for pos in self.dot_lists[oblast_name]["lepto"]:
                        pygame.draw.circle(self.dots_surface, COLOR_LEPTO_DOT, pos, DOT_RADIUS)
            
            self.map_surface.blit(self.dots_surface, (0, 0)) 
            
    def draw_ui_panel(self):
        self.ui_surface.fill(COLOR_UI_PANEL)
        pygame.draw.line(self.ui_surface, (200, 200, 200), (0, 0), (0, SCREEN_HEIGHT), 2)
        
        px, py = 20, 20 
        
        text = self.font_large.render("Налаштування", True, COLOR_TEXT)
        self.ui_surface.blit(text, (px, py))
        
        if self.app_state == "SETUP":
            text = self.font_medium.render("Тривалість (днів):", True, COLOR_TEXT)
            self.ui_surface.blit(text, (px, py + 40))
            self.input_boxes['duration'].draw(self.ui_surface)
            
            text = self.font_medium.render("Початковий день (1-365):", True, COLOR_TEXT)
            self.ui_surface.blit(text, (px, py + 100))
            self.input_boxes['start_day'].draw(self.ui_surface)
            
            py_oblast = py + 200
            if self.selected_oblast_name:
                text = self.font_large.render(self.selected_oblast_name, True, COLOR_TEXT)
                self.ui_surface.blit(text, (px, py_oblast))
                text = self.font_medium.render("Інф. Гризуни (I_R):", True, COLOR_TEXT)
                self.ui_surface.blit(text, (px, py_oblast + 40))
                self.input_boxes['oblast_ir'].draw(self.ui_surface)
                text = self.font_medium.render("Інф. Тварини (I_A):", True, COLOR_TEXT)
                self.ui_surface.blit(text, (px, py_oblast + 100))
                self.input_boxes['oblast_ia'].draw(self.ui_surface)
                text = self.font_medium.render("Забруднення (W):", True, COLOR_TEXT)
                self.ui_surface.blit(text, (px, py_oblast + 160))
                self.input_boxes['oblast_w'].draw(self.ui_surface)
            
            else:
                 text = self.font_medium.render("Клікніть на область...", True, (150, 150, 150))
                 self.ui_surface.blit(text, (px, py_oblast))

        py_status = SCREEN_HEIGHT - 200 
        
        if self.app_state == "SETUP":
            text = self.font_medium.render("Натисніть [ENTER] для запуску", True, (0, 100, 0))
            self.ui_surface.blit(text, (px, py_status+100))
            
        elif self.app_state == "SIMULATING":
            text = self.font_large.render("РОЗРАХУНОК...", True, (200, 0, 0))
            self.ui_surface.blit(text, (px, py_status))
            
        elif self.app_state == "ANIMATING":
            day_text = f"День: {int(self.current_day)}"
            if self.t_eval is not None:
                day_text += f" / {len(self.t_eval)-1}"
            
            text_day = self.font_large.render(day_text, True, COLOR_TEXT)
            self.ui_surface.blit(text_day, (px, py_status))
            
            if self.simulation_finished:
                text_end = self.font_large.render("Завершено", True, (0,100,0))
                self.ui_surface.blit(text_end, (px, py_status + 40)) 
                
                text_reset = self.font_medium.render(f"Скинути: [R]", True, (100,100,100))
                self.ui_surface.blit(text_reset, (px, py_status + 70))

                if self.graph_selected_oblast:
                    text_sel = self.font_medium.render(f"Обрано: {self.graph_selected_oblast}", True, COLOR_TEXT)
                    self.ui_surface.blit(text_sel, (px, py_status + 100))
                    
                    self.graph_button_rect = pygame.Rect(px, py_status + 125, SIM_PANEL_WIDTH - 40, 30)
                    pygame.draw.rect(self.ui_surface, COLOR_GRAPH_BUTTON, self.graph_button_rect, border_radius=5)
                    text_graph = self.font_medium.render("Показати Графік", True, (255, 255, 255))
                    text_rect = text_graph.get_rect(center=self.graph_button_rect.center)
                    self.ui_surface.blit(text_graph, text_rect)
                else:
                    self.graph_button_rect = None
                    text_click = self.font_medium.render("Клікніть на область для графіка", True, (100,100,100))
                    self.ui_surface.blit(text_click, (px, py_status + 100))

            else:
                text_speed = self.font_medium.render(f"Швидкість: x{self.playback_speed:.2f}", True, COLOR_TEXT)
                self.ui_surface.blit(text_speed, (px, py_status + 40))

                pause_text = "ПАУЗА" if self.is_paused else "ВІДТВОРЕННЯ"
                text_pause = self.font_medium.render(f"[{pause_text}] (Пробіл)", True, (100,100,100))
                self.ui_surface.blit(text_pause, (px, py_status + 70))

                text_speed_keys = self.font_medium.render(f"Швидкість: [<] [>]", True, (100,100,100))
                self.ui_surface.blit(text_speed_keys, (px, py_status + 90))
                
                if self.is_paused: 
                    text_reset = self.font_medium.render(f"Скинути: [R]", True, (100,100,100))
                    self.ui_surface.blit(text_reset, (px, py_status + 110))

    def draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        self.draw_map_surface()
        self.screen.blit(self.map_surface, (0, 0))
        self.draw_ui_panel()
        self.screen.blit(self.ui_surface, (MAP_WIDTH, 0))
        
        if self.hovered_oblast_name and (self.app_state == "SETUP" or self.app_state == "ANIMATING"):
            mouse_pos = pygame.mouse.get_pos()
            text_surf = self.font_medium.render(self.hovered_oblast_name, True, COLOR_TEXT)
            text_rect = text_surf.get_rect(topleft=(mouse_pos[0] + 15, mouse_pos[1] + 15))
            
            bg_rect = text_rect.inflate(10, 10)
            pygame.draw.rect(self.screen, COLOR_TOOLTIP_BG, bg_rect, border_radius=3)
            pygame.draw.rect(self.screen, COLOR_MAP_BORDER, bg_rect, 1, border_radius=3)
            
            self.screen.blit(text_surf, text_rect)

        pygame.display.flip()

    # [НОВА ФУНКЦІЯ]
    def show_graph(self):
        if not self.graph_selected_oblast or self.sim_results is None:
            print("Помилка: Немає даних для побудови графіка.")
            return

        name = self.graph_selected_oblast
        idx = self.sim_data['oblast_to_index'][name]
        days = self.t_eval
        
        hanta_humans = self.sim_results[idx, simulation_engine.i_I_H, :]
        lepto_humans = self.sim_results[idx, simulation_engine.i_I_L, :]
        
        hanta_rodents = self.sim_results[idx, simulation_engine.i_I_R, :]
        lepto_animals = self.sim_results[idx, simulation_engine.i_I_A, :]
        lepto_water = self.sim_results[idx, simulation_engine.i_W, :]
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(days, hanta_humans, label='Хантавірус (Люди)', color='blue')
        plt.plot(days, lepto_humans, label='Лептоспіроз (Люди)', color='red')
        plt.title(f"Динаміка інфекцій у людей: {name}")
        plt.xlabel('Дні')
        plt.ylabel('Частка інфікованих')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(days, hanta_rodents, label='Інф. Гризуни (Ханта)', color='blue', linestyle='--')
        plt.plot(days, lepto_animals, label='Інф. Тварини (Лепто)', color='red', linestyle='--')
        plt.plot(days, lepto_water, label='Забруднення (Лепто)', color='orange', linestyle=':')
        plt.title(f"Динаміка інфекцій у резервуарах: {name}")
        plt.xlabel('Дні')
        plt.ylabel('Умовні одиниці популяції/забруднення')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


    def run(self):
        if not self.load_data_and_map():
            return
        self.app_state = "SETUP" 
        
        while True:
            delta_time_ms = self.clock.tick(60) 
            self.handle_events()
            self.update(delta_time_ms) 
            self.draw()

if __name__ == '__main__':
    app = App()
    app.run()