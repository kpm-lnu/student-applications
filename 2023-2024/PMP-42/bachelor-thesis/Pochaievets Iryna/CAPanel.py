import pygame
from pygame import gfxdraw
from CA import CACell 
from CustomRoom import CustomRoom  
import time

# Колірні константи
BLUE = (0, 0, 255)  # Колір для синіх виходів
WHITE = (240, 240, 240)   # Колір для білої сітки
GRAY = (105, 105, 105)    # Колір для сірої сітки
BLACK = (10, 10, 10)      # Колір для чорних преп'ятствій
RED = (255, 0, 0)         # Колір для червоних виходів
BACKGROUND_COLOR = (100, 100, 100)  # Колір для фону

class SimuPanel:
    def __init__(self, room, cell_size_px):
        # Ініціалізує панель симуляції з заданим кімнатним об'єктом та розміром клітини
        self.room = room
        self.cell_size_px = cell_size_px
        self.simu_surf = pygame.Surface((self.room.nx * self.cell_size_px + 1, self.room.ny * self.cell_size_px + 1))
        self.grid_surf = pygame.Surface((self.room.nx * self.cell_size_px + 1, self.room.ny * self.cell_size_px + 1))
        self.grid_surf.fill(WHITE)  # Заповнює сітку білим кольором
        self._render_grid()  # Викликає внутрішній метод для відображення сітки

    def _render_grid(self):
        # Малює сітку на вікні симуляції
        x0, y0 = 0, 0
        cell_size = self.cell_size_px

        for i in range(self.room.ny):
            pygame.draw.line(self.grid_surf, GRAY, (x0, y0 + i * cell_size), (x0 + self.room.nx * cell_size, y0 + i * cell_size))
            for j in range(self.room.nx):
                pygame.draw.line(self.grid_surf, GRAY, (x0 + j * cell_size, y0), (x0 + j * cell_size, y0 + self.room.ny * cell_size))

        # Завершення правої та нижньої стін
        pygame.draw.line(self.grid_surf, GRAY, (x0, y0 + self.room.ny * cell_size), (x0 + self.room.nx * cell_size, y0 + self.room.ny * cell_size))
        pygame.draw.line(self.grid_surf, GRAY, (x0 + self.room.nx * cell_size, y0), (x0 + self.room.nx * cell_size, y0 + self.room.ny * cell_size))

    def render(self, surf, pos):
        # Рендерить відображення симуляції на вказаній поверхні (surf) у вказаному положенні (pos)
        cell_size = self.cell_size_px

        # Спочатку блокуємо пусту сітку
        self.simu_surf.blit(self.grid_surf, (0, 0))

        for i in range(self.room.ny):
            for j in range(self.room.nx):
                cell_ij = self.room.getCell((i, j))  # Отримує комірку у вказаних координатах

                if cell_ij.state == CACell.OBSTACLE_STATE:  # Якщо це перешкода, малюємо чорний прямокутник
                    pygame.draw.rect(self.simu_surf, GRAY, (j * cell_size, i * cell_size, cell_size, cell_size), 0)
                elif cell_ij.state == CACell.EXIT_STATE:  # Якщо це вихід, малюємо червоний прямокутник
                    pygame.draw.rect(self.simu_surf, RED, (j * cell_size, i * cell_size, cell_size, cell_size), 0)
                elif cell_ij.state == CACell.PERSON_STATE:  # Якщо це людина, малюємо коло з вказаним кольором
                    fcolor = cell_ij.properties['fill']
                    gfxdraw.aacircle(self.simu_surf, int(j * cell_size + cell_size / 2), int(i * cell_size + cell_size / 2), int(cell_size / 4), fcolor)
                    gfxdraw.filled_circle(self.simu_surf, int(j * cell_size + cell_size / 2), int(i * cell_size + cell_size / 2), int(cell_size / 4), fcolor)
                elif cell_ij.state == CACell.EMPTY_STATE:  # Якщо комірка порожня, пропускаємо
                    pass

        surf.blit(self.simu_surf, pos)  # Відображаємо панель симуляції на поверхні (surf) у вказаному положенні (pos)

def handle_events():
    # Обробляє події Pygame, такі як вихід з програми
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True


def main():
    pygame.init()
    SCREEN_W, SCREEN_H = 550, 550
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption('Evacuation')
    
    canvas = pygame.Surface((SCREEN_W, SCREEN_H))
    canvas.fill(BACKGROUND_COLOR)
    
    begin = time.time()

    room, people = CustomRoom.make_3_exits_room(full_factor=0.67, pos_seed=7, panic_prob=0.3)
    simu_panel = SimuPanel(room, 8)
    
    FPS = 30
    evac_total = 0
    iteration = 0
    running = True
    clock = pygame.time.Clock()

    while running:
        clock.tick(FPS)
        running = handle_events()
        
        evac = room.step_parallel()
        evac_total += evac
        iteration += 1
        #print(f'iter {iteration}: (+{evac}) {evac_total}/{people}')
        
        if people == evac_total:
            running = False

        # Рендер
        canvas.fill(BACKGROUND_COLOR)
        simu_panel.render(canvas, (10, 10))
        screen.blit(canvas, (0, 0))
        pygame.display.flip()
    end = time.time()
    print(end-begin)
    pygame.quit()

if __name__ == '__main__':
    main()
