import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.patches as patches
from shapely.geometry import Polygon
from planner import ObstacleAvoidancePathPlanner
import time
from datetime import datetime


class InteractiveMapCreator:
    def __init__(self):
        self.start_time = time.time()
        print(f"Програма запущена о {datetime.now().strftime('%H:%M:%S')}")
        
        self.stage = 0  
        self.map_size = 10
        self.num_points = 200
        self.start_point = None
        self.goal_point = None
        self.obstacles = []
        self.current_obstacle = []

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.3)

        self.start_marker = None
        self.goal_marker = None
        self.obstacle_patches = []

        self.setup_ui()
        self.setup_event_handlers()

        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.update_title()
        plt.show()

    def setup_ui(self):
        self.text_boxes = {}

        ax_size = plt.axes([0.2, 0.2, 0.1, 0.05])
        self.text_boxes['size'] = TextBox(ax_size, 'Розмір карти:', initial=str(self.map_size))

        ax_points = plt.axes([0.4, 0.2, 0.1, 0.05])
        self.text_boxes['points'] = TextBox(ax_points, 'Кількість точок:', initial=str(self.num_points))
        self.text_boxes['points'].set_active(False)

        ax_clear = plt.axes([0.6, 0.2, 0.1, 0.05])
        self.clear_btn = Button(ax_clear, 'Очистити')

        ax_next = plt.axes([0.8, 0.2, 0.1, 0.05])
        self.next_btn = Button(ax_next, 'Далі')

    def setup_event_handlers(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.clear_btn.on_clicked(self.clear_map)
        self.next_btn.on_clicked(self.next_stage)
        self.text_boxes['size'].on_submit(self.update_map_size)
        self.text_boxes['points'].on_submit(self.update_num_points)

    def update_title(self):
        titles = [
            "Крок 1: Введіть розмір карти та натисніть 'Далі'",
            "Крок 2: ЛКМ — додати точку перешкоди, ПКМ — завершити перешкоду",
            "Крок 3: ЛКМ — встановити старт та фініш (спочатку старт, потім фініш)",
            "Крок 4: Введіть кількість точок, натисніть 'Далі' для побудови маршруту"
        ]
        self.ax.set_title(titles[self.stage])
        self.fig.canvas.draw_idle()

    def update_map_size(self, text):
        if self.stage == 0:
            try:
                self.map_size = float(text)
                self.ax.set_xlim(0, self.map_size)
                self.ax.set_ylim(0, self.map_size)
                self.fig.canvas.draw_idle()
            except ValueError:
                pass

    def update_num_points(self, text):
        if self.stage == 3:
            try:
                self.num_points = int(text)
            except ValueError:
                pass

    def next_stage(self, event):
        if self.stage == 3:
            self.run_planner()
        else:
            self.stage += 1
            if self.stage == 3:
                self.text_boxes['points'].set_active(True)
            self.update_title()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if self.stage == 1:
            if event.button == 1:
                self.current_obstacle.append((x, y))
                self.draw_current_obstacle()
            elif event.button == 3 and len(self.current_obstacle) >= 3:
                self.obstacles.append(self.current_obstacle)
                self.finalize_obstacle()
                self.current_obstacle = []

        elif self.stage == 2:
            if event.button == 1:
                point = (x, y)
                from shapely.geometry import Point
                from shapely.ops import unary_union
                all_obstacles = unary_union([Polygon(o) for o in self.obstacles])
                if all_obstacles.contains(Point(point)):
                    print("Точка не може бути всередині перешкоди.")
                    return
                if self.start_point is None:
                    self.start_point = point
                    print(f"Стартова точка встановлена: ({x:.2f}, {y:.2f})")
                elif self.goal_point is None:
                    self.goal_point = point
                    print(f"Фінішна точка встановлена: ({x:.2f}, {y:.2f})")
                self.draw_start_goal()

    def draw_current_obstacle(self):
        for patch in self.ax.patches:
            if patch.get_label() == 'temp_obstacle':
                patch.remove()

        if len(self.current_obstacle) >= 2:
            polygon = patches.Polygon(self.current_obstacle, closed=False, fill=False, color='red', alpha=0.5, label='temp_obstacle')
            self.ax.add_patch(polygon)
            self.fig.canvas.draw_idle()

    def finalize_obstacle(self):
        polygon = patches.Polygon(self.current_obstacle, closed=True, fill=True, color='red', alpha=0.5, label='Перешкоди')
        self.ax.add_patch(polygon)
        self.obstacle_patches.append(polygon)
        
        print(f"\nДодано перешкоду {len(self.obstacles)+1} з координатами:")
        for i, (x, y) in enumerate(self.current_obstacle):
            print(f"Точка {i+1}: ({x:.2f}, {y:.2f})")
        
        self.draw_start_goal()
        self.fig.canvas.draw_idle()

    def draw_start_goal(self):
        if self.start_marker:
            self.start_marker.remove()
        if self.goal_marker:
            self.goal_marker.remove()

        if self.start_point:
            self.start_marker = self.ax.plot(*self.start_point, 'yo', markersize=12, label='Стартова точка')[0]
        if self.goal_point:
            self.goal_marker = self.ax.plot(*self.goal_point, 'mo', markersize=12, label='Кінцева точка')[0]

        if self.obstacle_patches:
            obstacle_patch = patches.Patch(color='red', alpha=0.5, label='Перешкоди')
            handles = [obstacle_patch]
            if self.start_marker:
                handles.append(self.start_marker)
            if self.goal_marker:
                handles.append(self.goal_marker)
            self.ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5))

        self.fig.canvas.draw_idle()

    def clear_map(self, event):
        self.ax.clear()
        self.stage = 0
        self.map_size = 10
        self.num_points = 200
        self.start_point = None
        self.goal_point = None
        self.obstacles = []
        self.current_obstacle = []
        self.start_marker = None
        self.goal_marker = None
        self.obstacle_patches = []
        self.text_boxes['size'].set_val(str(self.map_size))
        self.text_boxes['points'].set_val(str(self.num_points))
        self.text_boxes['points'].set_active(False)

        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.update_title()
        print("\nКарта очищена. Початок роботи з новою картою.")

    def run_planner(self):
        if not self.start_point or not self.goal_point:
            print("Будь ласка, вкажіть стартову та фінішну точки")
            return

        print("\n=== Інформація про карту ===")
        print(f"Розмір карти: {self.map_size}x{self.map_size}")
        print(f"Кількість точок для триангуляції: {self.num_points}")
        print(f"Стартова точка: ({self.start_point[0]:.2f}, {self.start_point[1]:.2f})")
        print(f"Фінішна точка: ({self.goal_point[0]:.2f}, {self.goal_point[1]:.2f})")
        print("\nСписок усіх перешкод:")
        for i, obstacle in enumerate(self.obstacles, 1):
            print(f"\nПерешкода {i}:")
            for j, (x, y) in enumerate(obstacle):
                print(f"Точка {j+1}: ({x:.2f}, {y:.2f})")

        print("\nПочаток побудови маршруту...")
        start_planning_time = time.time()
        
        planner = ObstacleAvoidancePathPlanner(
            self.start_point,
            self.goal_point,
            self.obstacles,
            map_size=self.map_size,
            num_points=self.num_points
        )
        
        path, points, triangles, path_exists, total_distance = planner.plan_path()
        planning_time = time.time() - start_planning_time
        print(f"\nЧас побудови маршруту: {planning_time:.2f} секунд")
        
        if path_exists:
            print("\nМаршрут успішно знайдено!")
            print("Точки маршруту:")
            for i, (x, y) in enumerate(path):
                print(f"{i+1}. ({x:.2f}, {y:.2f})")
            print(f"\nЗагальна довжина маршруту: {total_distance:.2f} одиниць")
        else:
            print("\nНе вдалося знайти маршрут!")
        
        total_time = time.time() - self.start_time
        print(f"\nЗагальний час виконання: {total_time:.2f} секунд")
        
        planner.visualize(path, points, triangles, path_exists, total_distance)