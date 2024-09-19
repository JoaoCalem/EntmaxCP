from PIL import Image
import os
corrupted_files = []
folders = os.listdir('./data/imagenet/train')
folders.remove('.DS_Store')
for folder in folders:
    for file in os.listdir(f'./data/imagenet/train/{folder}'):
        filename = f'./data/imagenet/train/{folder}/{file}'
        try:
            im = Image.open(filename)
            im.verify() #I perform also verify, don't know if he sees other types o defects
            im.close() #reload is necessary in my case
            im = Image.open(filename) 
            im.transpose(Image.FLIP_LEFT_RIGHT)
            im.close()
        except:   
            corrupted_files.append(filename)
            
print(len(corrupted_files))
with open("corrupted.txt", "w") as output:
    output.write(str(corrupted_files))