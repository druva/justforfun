import os
import random
from PIL import Image, ImageDraw

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def createShape(shape, filename):
    # Image dimensions
    WIDTH, HEIGHT = 100, 100
    if shape == 'rectangle':
        HEIGHT = 80

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    x = random.randint(0, WIDTH // 2)
    y = random.randint(0, HEIGHT // 2)
    w = random.randint(30, WIDTH - x)
    h = random.randint(30, HEIGHT - y)

    outline_color = random_color()
    fill_color = random_color()

    if shape == 'circle':
        size = min(w, h)
        bounds = (x, y, x + size, y + size)
        draw.ellipse(bounds, outline=outline_color, width=5, fill=fill_color)

    elif shape == 'square':
        size = min(w, h)
        bounds = (x, y, x + size, y + size)
        draw.rectangle(bounds, outline=outline_color, width=5, fill=fill_color)

    elif shape == 'rectangle':
        bounds = (x, y, x + w, y + h)
        draw.rectangle(bounds, outline=outline_color, width=5, fill=fill_color)

    elif shape == 'triangle':
        point1 = (x + w // 2, y)
        point2 = (x, y + h)
        point3 = (x + w, y + h)
        draw.polygon([point1, point2, point3], outline=outline_color, fill=fill_color)

    image.save(filename, "JPEG")
    print(f"Image saved as {filename}")

shapes = ["circle", "square", "rectangle", "triangle"]
phases = ["train", "validation"]

base_dir = "images"
shape_dirs = {
    f"{shape}_{phase}": os.path.join(base_dir, phase, shape)
    for shape in shapes
    for phase in phases
}

# Create directories
for path in shape_dirs.values():
    os.makedirs(path, exist_ok=True)

def createTestSet(max):
    ti = max - (max // 10)
    for shape in shapes:
        for i in range(1, max + 1):
            fol = 'train' if i <= ti else 'validation'
            filename = os.path.join(shape_dirs[f'{shape}_{fol}'], f'{shape}_{i}.jpg')
            createShape(shape, filename)

createTestSet(1000)
