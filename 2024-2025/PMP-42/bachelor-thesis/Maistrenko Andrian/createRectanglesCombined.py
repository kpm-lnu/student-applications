import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

def write_quad4_file(fname, quad4_list):
    """Запис Quad4 у файл."""
    with open(fname, 'w') as f:
        for quad4 in quad4_list:
            line = " ".join(f"({x:.6f}, {y:.6f})" for x, y in quad4)
            f.write(line + "\n")

def generate_rectangle(height):
    """Генерує прямокутник з фіксованою основою (1,0)-(2,0) та заданою висотою."""
    return [
        (1.0, 0.0),    # нижній лівий
        (2.0, 0.0),    # нижній правий
        (2.0, height), # верхній правий
        (1.0, height)  # верхній лівий
    ]

def generate_rectangles_q4(n):
    rectangles = []
    for _ in range(n):
        if random.random() < 0.5:
            height = random.uniform(0.1, 0.8)
        else:
            height = random.uniform(2.0, 64.0)
        rect4 = generate_rectangle(height)
        rectangles.append(rect4)
    return rectangles

def is_convex(polygon):
    n = len(polygon)
    if n < 3:
        return True
    cp0 = None
    for i in range(n):
        p1, p2, p3 = polygon[i], polygon[(i+1) % n], polygon[(i+2) % n]
        cp = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        if cp0 is None:
            cp0 = cp
        elif cp0 * cp < 0:
            return False
    return True

def generate_convex_quad():
    fixed = [(1.0, 0.0), (2.0, 0.0)]
    while True:
        rnd = [(random.uniform(0.5, 6), random.uniform(0.5, 6)) for _ in range(2)]
        pts = fixed + rnd
        for perm in itertools.permutations(pts):
            if perm[0] == fixed[0] and perm[1] == fixed[1] and is_convex(perm):
                return list(perm)

def generate_quads_q4(n):
    quads = []
    while len(quads) < n:
        quad4 = generate_convex_quad()
        quads.append(quad4)
    return quads

def visualize_shapes(rectangles, quads, num_to_show=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    ax1.set_title("Прямокутники (Q4)", fontsize=14)
    ax1.set_xlim(0.5, 2.5)
    ax1.set_ylim(-0.5, 20)
    ax1.grid(True, alpha=0.3)
    for rect4 in rectangles[:num_to_show]:
        width = rect4[1][0] - rect4[0][0]
        height = rect4[2][1] - rect4[1][1]
        ax1.add_patch(patches.Rectangle(
            (rect4[0][0], rect4[0][1]), width, height,
            edgecolor='blue', facecolor='blue', alpha=0.4
        ))
        ax1.plot([1.0, 2.0], [0.0, 0.0], 'ro', markersize=8, label='Фіксовані точки' if rect4 == rectangles[0] else "")
    
    ax2.set_title("Опуклі чотирикутники (Q4)", fontsize=14)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 3)
    ax2.grid(True, alpha=0.3)
    for quad4 in quads[:num_to_show]:
        ax2.add_patch(patches.Polygon(
            quad4, closed=True,
            edgecolor='red', facecolor='red', alpha=0.4
        ))
        ax2.plot([1.0, 2.0], [0.0, 0.0], 'ro', markersize=8, label='Фіксовані точки' if quad4 == quads[0] else "")
    
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()

def print_statistics(rectangles, quads):
    """Статистика для обох типів."""
    r_heights = [rect[2][1] for rect in rectangles]
    q_heights = [max(q[2][1], q[3][1]) for q in quads]  
    
    print("📊 Статистика:")
    print(f"Прямокутники: {len(rectangles)} (h: {min(r_heights):.1f}-{max(r_heights):.1f})")
    print(f"Чотирикутники: {len(quads)} (приблизна висота: {min(q_heights):.1f}-{max(q_heights):.1f})")
    print(f"Всі елементи мають фіксовані точки: (1.0, 0.0) та (2.0, 0.0)")

if __name__ == "__main__":
    n_total = 20000
    n_rect = n_total // 2
    n_quad = n_total - n_rect
    
    rectangles = generate_rectangles_q4(n_rect)
    quads = generate_quads_q4(n_quad)
    combined = rectangles + quads
    random.shuffle(combined)  
    
    write_quad4_file("combined_shapes_q4.txt", combined)
    print(f"✅ Згенеровано: {len(combined)} Q4 елементів (з них {n_rect} прямокутників).")
    
    print_statistics(rectangles, quads)
    visualize_shapes(rectangles, quads)