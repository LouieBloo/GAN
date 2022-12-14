from PIL import Image
import glob
import random

image_path="D:\Downloads\Images\dataset-part1\*.png"
dest_path = "D:\Downloads\Images\cat_holed"

hole_size = 20

# train_ds = keras.preprocessing.image_dataset_from_directory(
#     directory=image_path,
#     label_mode=None, image_size=(64, 64), batch_size=60000,
#     shuffle=False, seed=None
# )


# def save_image(image,name):
#     img = keras.preprocessing.image.array_to_img(image)
#     img.save(name.replace("dataset-part1","cat_holed"))


for filename in glob.glob(image_path): #assuming gif
    img = Image.open(filename)
    img_pixels = img.load()


    x_start = random.randint(0, 64-hole_size)
    x_end = x_start + hole_size

    y_start = random.randint(0, 64-hole_size)
    y_end = y_start + hole_size

    for i in range(img.size[0]): # for every pixel:
        for j in range(img.size[1]):
            if i >= x_start and i < x_end and j >= y_start and j < y_end:
                img_pixels[i,j] = (255,255 ,255)

    img.save(filename.replace("dataset-part1","cat_holed"))

# #tr = train_ds.as_numpy_iterator()
# file_paths = train_ds.file_paths
# for batch in train_ds:
#     index = 0
#     for pic in batch:
#         if index < 10:
#             img = batch[index]
            
#             for x in range(63):
#                 img[x] = 0

#             save_image(batch[index],file_paths[index])
#         index += 1
    