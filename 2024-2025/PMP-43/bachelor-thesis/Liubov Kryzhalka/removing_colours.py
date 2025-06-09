from PIL import Image

img = Image.open('./photos/1.jpg')
rgba = img.convert("RGBA")
datas = rgba.getdata()

newData = []

def is_in_range(color, ranges):
    r, g, b = color
    return all(low <= val <= high for val, (low, high) in zip((r, g, b), ranges))

blue_range = ((0, 220), (90, 240), (135, 255))    # блакитний
red_range = ((0, 255), (0, 90), (0, 90))          # червоний
brown_range = ((100, 180), (50, 100), (30, 70))   # коричневий

for item in datas:
    if (is_in_range(item[:3], blue_range) or
        is_in_range(item[:3], red_range) or
        is_in_range(item[:3], brown_range)):
        newData.append((255, 255, 255, 0))  # прозорий
    else:
        newData.append(item)

rgba.putdata(newData)
rgba.save("./photos/2.png", "PNG")
