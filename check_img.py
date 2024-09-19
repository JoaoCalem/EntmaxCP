from PIL import Image
import os

for folder in os.listdir('./data/imagenet/train'):
    for file in os.listdir(f'./data/imagenet/train/{folder}'):
        filename = f'./data/imagenet/train/{folder}/{file}'
        try:
            im = Image.load(filename)
            im.verify() #I perform also verify, don't know if he sees other types o defects
            im.close() #reload is necessary in my case
            im = Image.load(filename) 
            im.transpose(Image.FLIP_LEFT_RIGHT)
            im.close()
        except:   
            print(filename)