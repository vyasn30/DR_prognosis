import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import cv2

#
# def resize_image(img_path, desired_size = 224):
#     im = Image.open(img_path)
#     im = im.resize((desired_size, ) * 2, resample = Image.LANCZOS)
#     return im
#
#
#
#
# train_df = pd.read_csv("data/aptos2019-blindness-detection/train.csv")
# test_df = pd.read_csv("data/aptos2019-blindness-detection/test.csv")
#
# print(train_df.shape)
# print(test_df.shape)
#
# N_train = train_df.shape[0]
# N_test = test_df.shape[0]
#
# x_train = np.empty((N_train, 224, 224, 3), dtype = np.uint8)
# x_test = np.empty((N_test, 224, 224, 3), dtype = np.uint8)
#
# for i, image_id in enumerate(tqdm(train_df['id_code'])):
#     x_train[i, :, :, :] = resize_image(
#         'data/aptos2019-blindness-detection/train_images/' + image_id + ".png"
#     )
#
# for i, image_id in enumerate(tqdm(test_df['id_code'])):
#     x_test[i, :, :, :] = resize_image(
#         'data/aptos2019-blindness-detection/test_images/' + image_id + ".png"
#     )
#
# y_train = pd.get_dummies(train_df['diagnosis']).values
#
#
# np.save("x_train.npy", x_train)
# np.save("x_test.npy", x_test)
# np.save("y_train.npy", y_train)
#
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)

IMG_SIZE = 512

df_train = pd.read_csv("data/aptos2019-blindness-detection/train.csv")
df_test = pd.read_csv("data/aptos2019-blindness-detection/test.csv")

x = df_train["id_code"]
y = df_train["diagnosis"]

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size= 0.15, stratify=y)

print(train_x.head())

print(train_y.head())


fig = plt.figure(figsize=(25, 16))
for class_id in sorted(train_y.unique()):
    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5).iterrows()):
        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
        path=f"data/aptos2019-blindness-detection/train_images/{row['id_code']}.png"
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128) # the trick is to add this line

        plt.imshow(image, cmap='gray')
        #plt.show()
        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )
