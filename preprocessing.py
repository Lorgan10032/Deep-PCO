import os

from PIL import Image, ImageFilter, ImageEnhance


def preprocessing(dataset_dir):
    items = os.listdir(dataset_dir)
    image_paths = []
    for item in items:
        path = os.path.join(dataset_dir, item)
        if os.path.isfile(path):
            image_paths.append(path)

    # for image_path in image_paths:
    #     img = Image.open(image_path)
    #     img = img.convert('L')
    #     img = img.resize((500, 500))
    #     enhancer = ImageEnhance.Contrast(img)
    #     img = enhancer.enhance(1.5)

    # TODO
    return
