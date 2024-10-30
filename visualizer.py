#%%
import pandas as pd

annotations = pd.read_csv("runs/full_model/report.csv")

print(len(annotations))
annotations.head()

#%%

eyes_annotations = annotations.loc[annotations['detection'] == 'eye']
eyes_annotations = eyes_annotations.drop(columns=['detection'])

# two eyes share the same image_name, left tag goes to the one with the smallest x1 value and viceversa, create a new column to identify the eye
eyes_annotations['eye'] = eyes_annotations.groupby('image_name')['x1'].rank(method='dense', ascending=True)

# replace the eye column with left and right
eyes_annotations['eye'] = eyes_annotations['eye'].apply(lambda x: 'left' if x == 1 else 'right')

eyes_annotations.head()


#%%
# plot images with the eyes annotated
import matplotlib.pyplot as plt
import cv2

import numpy as np
from utils import smart_resize
images_path = "images/frame_filmato0"

# pick 5 random images
# images = np.random.choice(eyes_annotations['image_name'].unique(), 5)
images = eyes_annotations['image_name'].unique()[:20]



for image_name in images:

    # image was resized to 640x640 for the model training so we need to resize the annotations
    
    img = cv2.imread(f"{images_path}/{image_name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_annotations = eyes_annotations.loc[eyes_annotations.loc[:, 'image_name'] == image_name].copy()
    
    ratio = img.shape[1] / 640

    img_annotations['x1'] = (img_annotations['x1'] * ratio).astype(int)
    img_annotations['x2'] = (img_annotations['x2'] * ratio).astype(int)
    img_annotations['y1'] = (img_annotations['y1'] * ratio).astype(int)
    img_annotations['y2'] = (img_annotations['y2'] * ratio).astype(int)


    

    plt.imshow(img)
    for i, row in img_annotations.iterrows():
        x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']]
        eye = row['eye']
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label=eye)
    plt.show()

#%%
# for image in images, crop the eyes and save them
output_path = "images/eyes_crop"

import os
os.makedirs(output_path, exist_ok=True)

for image_name in images:

    # image was resized to 640x640 for the model training so we need to resize the annotations
    
    img = cv2.imread(f"{images_path}/{image_name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_annotations = eyes_annotations.loc[eyes_annotations.loc[:, 'image_name'] == image_name].copy()
    
    ratio = img.shape[1] / 640

    img_annotations['x1'] = (img_annotations['x1'] * ratio).astype(int)
    img_annotations['x2'] = (img_annotations['x2'] * ratio).astype(int)
    img_annotations['y1'] = (img_annotations['y1'] * ratio).astype(int)
    img_annotations['y2'] = (img_annotations['y2'] * ratio).astype(int)

    for i, row in img_annotations.iterrows():
        x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']]
        eye = row['eye']

        # add some padding
        padding = 10
        x1 = max(0, x1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y1 = max(0, y1 - padding)
        y2 = min(img.shape[0], y2 + padding)

        eye_img = img[y1:y2, x1:x2]
        cv2.imwrite(f"{output_path}/{image_name}_{eye}.bmp", cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR))




#%%

import time
import os
import cv2

eyes = [f for f in os.listdir(output_path) if 'left' in f]
eyes.sort()

# create a video with the eyes cropped

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('eyes.avi', fourcc, 1, (120, 65))

for eye in eyes:
    img = cv2.imread(f"{output_path}/{eye}")
    img = cv2.resize(img, (120, 65))
    out.write(img)

out.release()



#%%
from utils import smart_resize

img = cv2.imread("images/frame_filmato0/0001.bmp")
print(img.shape)

img, _ = smart_resize(img, new_size=640)
print(img.shape)
