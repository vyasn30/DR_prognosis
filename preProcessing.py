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
def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance

    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


# OLD version of image color cropping, use crop_image_from_gray instead
# The above code work only for 1-channel. Here is my simple extension for 3-channels image
def crop_image(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        h, w, _ = img.shape
        #         print(h,w)
        img1 = cv2.resize(crop_image1(img[:, :, 0]), (w, h))
        img2 = cv2.resize(crop_image1(img[:, :, 1]), (w, h))
        img3 = cv2.resize(crop_image1(img[:, :, 2]), (w, h))

        #         print(img1.shape,img2.shape,img3.shape)
        img[:, :, 0] = img1
        img[:, :, 1] = img2
        img[:, :, 2] = img3
        return img


'''all of these do not work'''


def crop_image2(image, threshold=5):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def crop_image3(image):
    mask = image > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = image[x0:x1, y0:y1]
    return cropped


def crop_image4(image):
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y + h, x:x + w]
    return crop


def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image

def circle_crop(img, sigmaX=10):
    """
    Create circular crop around image centre
    """

    img = cv2.imread(img)
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img


IMG_SIZE = 512

df_train = pd.read_csv("data/aptos2019-blindness-detection/train.csv")
df_test = pd.read_csv("data/aptos2019-blindness-detection/test.csv")

x = df_train["id_code"]
y = df_train["diagnosis"]

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size= 0.15, stratify=y)

print(train_x.head())

print(train_y.head())


NUM_SAMP=7
fig = plt.figure(figsize=(25, 16))
for class_id in sorted(train_y.unique()):
    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(NUM_SAMP).iterrows()):
        ax = fig.add_subplot(5, NUM_SAMP, class_id * NUM_SAMP + i + 1, xticks=[], yticks=[])
        path=f"data/aptos2019-blindness-detection/train_images/{row['id_code']}.png"
        image = load_ben_color(path,sigmaX=30)

        plt.imshow(image)
        plt.show()
        ax.set_title('%d-%d-%s' % (class_id, idx, row['id_code']) )
