from CA import CARoom, CACell
from itertools import product
from random import Random

class CustomRoom(object):
    @staticmethod
    def make_3_exits_room(full_factor=0.67, pos_seed=None, panic_prob=0.01, exit_size=3):
        room_w = 20
        room_h = 20
        room = CARoom(room_w, room_h)

        # Встановлення зовнішніх стін зверху та знизу
        for row in [0, room.ny - 1]:
            for col in range(room.nx):
                room.setCell((row, col), CACell(CACell.OBSTACLE_STATE, {}))

        # Встановлення зовнішніх стін зліва та справа
        for col in [0, room.nx - 1]:
            for row in range(room.ny):            
                room.setCell((row, col), CACell(CACell.OBSTACLE_STATE, {}))

        """
        # Виходи
        #три виходи
        exits = [(0, room.nx // 2 - exit_size // 2), (room.ny // 2 - exit_size // 2, room.nx - 1), (room.ny - 1, room.nx // 2 - exit_size // 2)]
        for ex in exits:
            for i in range(exit_size):
                if 0 <= ex[0] < room.ny and 0 <= ex[1] + i < room.nx:
                    room.setCell((ex[0], ex[1] + i), CACell(CACell.EXIT_STATE, {}))
		"""

		#Один вихід
        half_rows = int(room.ny/2)
        for i in range(exit_size):    
            room.setCell((half_rows + i, room.nx - 1), CACell(CACell.EXIT_STATE, {}))
		
        # Перша перешкода
        obstacle_size = 8
        col = int(room.nx * 6/10)
        row = 1 + int(room.ny/3) - obstacle_size

        row_cols = product(range(row, row + obstacle_size), range(col, col + obstacle_size))
        for r_c in row_cols:    
            room.setCell(r_c, CACell(CACell.OBSTACLE_STATE, {}))

        # Друга перешкода
        row = int(room.ny) - obstacle_size - row
        row_cols = product(range(row, row + obstacle_size), range(col, col + obstacle_size))

        for r_c in row_cols:
            room.setCell(r_c, CACell(CACell.OBSTACLE_STATE, {}))

        # Перешкода у вигляді стіни посередині при 1ому виході
        for c in range(col, col + obstacle_size):
            room.setCell((half_rows, c), CACell(CACell.OBSTACLE_STATE, {}))
		
        """
		# Перешкода у вигляді стіни посередині при 3 
        for c in range(col, col + obstacle_size):
            room.setCell((int(room.ny / 2), c), CACell(CACell.OBSTACLE_STATE, {}))
        """
 # Заповнення вільних клітинок людьми
        people = 0
        max_people = 500  # Максимальна кількість людей
        r = Random(pos_seed)

        for i in range(room.ny):
            for j in range(room.nx):
                if people >= max_people:
                    break
                if room.getCell((i, j)).state == CACell.EMPTY_STATE and r.random() < full_factor:
                    fill = (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255))
                    room.setCell((i, j), CACell(CACell.PERSON_STATE, {'fill': fill, 'panic_prob': panic_prob}))
                    people += 1

        return room, people
