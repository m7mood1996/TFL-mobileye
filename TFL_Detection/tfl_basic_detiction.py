try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse
    import cv2

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    actual_image = c_image

    c_image = c_image / 255
    c_image = c_image.astype(float)
    # c_image = c_image[:, :, [1]]
    blurred = ndimage.gaussian_filter(c_image[:, :, [1]], 3)
    filter_blurred = ndimage.gaussian_filter(blurred, 10)
    alpha = 30
    sharpened = blurred + alpha * (blurred - filter_blurred)
    m = sharpened.max()
    sharpened = sharpened / m
    kernel = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0.7, 0.7, 0],
                       [0, 0.7, 0.7, 0.7, 0.7],
                       [0, 0.7, 0.7, 0.7, 0.7]])
    kernel = np.resize(kernel, (7, 5, 1))

    kernel = kernel.astype(float)
    # kernel = np.fliplr(kernel)
    # kernel = np.flipud(kernel)

    kernel = kernel - np.average(kernel)

    t = sg.convolve(sharpened, kernel, mode='same')

    filterd_image = ndimage.maximum_filter(t, size=300)
    filterd_image = filterd_image == t
    green_j, green_i, z = np.where(filterd_image)
    i = 0
    green_j_list = []
    green_i_list = []
    for power in z:
        # if actual_image[green_j[i], green_i[i], 1] > 150 and actual_image[green_j[i], green_i[i], 0] < 160 :
        green_j_list.append(green_j[i])
        green_i_list.append(green_i[i])

        i += 1

    c_image = actual_image[:, :, [0]]
    c_image = c_image / 255
    blurred = ndimage.gaussian_filter(c_image[:, :, [0]], 3)
    filter_blurred = ndimage.gaussian_filter(blurred, 10)
    alpha = 30
    sharpened = blurred + alpha * (blurred - filter_blurred)
    m = sharpened.max()
    sharpened = sharpened / m
    kernel = np.fliplr(kernel)
    kernel = np.flipud(kernel)
    t = sg.convolve(sharpened, kernel, mode='same')

    filterd_image = ndimage.maximum_filter(t, size=300) == t
    red_j, red_i, z = np.where(filterd_image)
    i = 0
    red_j_list = []
    red_i_list = []
    for power in z:
        # if (actual_image[red_j[i],red_i[i],0] > 180 and 160 > actual_image[red_j[i],red_i[i],1]):
        red_j_list.append(red_j[i])
        red_i_list.append(red_i[i])
        i += 1

    print("pictur points :", len(red_i_list) + len(green_i_list))
    return red_i_list, red_j_list, green_i_list, green_j_list


def get_tfl_kernal(color, argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '/Users/mahmoodnael/PycharmProjects/scaleup'
    args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*tfl_' + color + '_.png'))
    tfl = []
    for image in flist:
        json_fn = image.replace('_.png', '_.json')
        if not os.path.exists(json_fn):
            json_fn = None
            tfl.append((np.array(Image.open(image)), image))
    return tfl


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def verify_tfl_images(red_x, red_y, green_x, green_y, image):
    tfl_positive = []
    tfl_nigative = []


    # if red tfl is positive
    i = 0
    while i < len(red_x):

        temp_img = image[red_y[i] - 40:red_y[i] + 41, red_x[i] - 40 : red_x[i] + 41 ]
        if temp_img.shape == (81, 81, 3):
            tfl_positive.append((red_x[i], red_y[i], temp_img))

        i += 1

    i = 0

    while i < len(green_x):

        temp_img = image[green_y[i] - 40:green_y[i] + 41, green_x[i] - 40: green_x[i] + 41]
        if temp_img.shape == (81,81, 3):
            tfl_nigative.append((green_x[i], green_y[i], temp_img))


        i += 1

    return tfl_positive, tfl_nigative


def show_croped_images(tfl_positive, tfl_nigative):
    w = 81
    h = 81
    fig = plt.figure()
    rows = 4
    cols = 4
    i = 0
    j = 1
    axes = []
    for tfl in tfl_positive:
        #plt.figure(i)

        axes.append(fig.add_subplot(rows, cols, j))
        subplot_title = ("YES TFL")
        axes[-1].set_title(subplot_title)
        plt.imshow(tfl)
        j += 1
        i += 1

    i = 0
    for tfl in tfl_nigative:

        axes.append(fig.add_subplot(rows, cols, j))
        subplot_title = ("NO TFL")
        axes[-1].set_title(subplot_title)

        plt.imshow(tfl)
        i += 1
        j += 1
    fig.tight_layout()
    plt.show()



def save_tfl_as_binary_files(tfl_positive, tfl_nigative, image_name):
    print(image_name)
    tfl_positive.tofile(image_name + "positive.bin")
    tfl_nigative.tofile(image_name + "nigative.bin")
    pass


def test_find_tfl_lights(image_path, json_path=None):
    """
    Run the attention code
    """

    image = np.array(Image.open(image_path))

    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]


    #show_image_and_gt(image, objects)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)



    tfl_positive, tfl_nigative = verify_tfl_images(red_x, red_y, green_x, green_y , image)


    # print(tfl_nigative)
    # print("nigative size", tfl_nigative.size)
    # save_tfl_as_binary_files(tfl_positive, tfl_nigative, image_name)

    #show_croped_images(tfl_positive, tfl_nigative) #function to plot trafic light

    #show_image_and_gt(image, objects)
    #plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    #plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
    # print(tfl_nigative)
    # print(tfl_positive)
    return tfl_positive, tfl_nigative


def save_tfl_data_as_binary(data, labeled):
    data.tofile("data.bin")
    labeled.tofile("labels.bin")

def main_tfl_basic_detiction(frame_path):
    DATA_PATH = "/Users/mahmoodnael/Desktop/Data"
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""


    positive = np.array([]).astype('uint8')
    negative = np.array([]).astype('uint8')

    po, ne = test_find_tfl_lights(frame_path)


    data = po + ne
    labeled = [1] * len(po) + [0] * len(ne)
    #labeled = np.append(np.ones(shape=(int(positive.size/(81*81*3))), dtype='uint8'),np.zeros(shape=(int(negative.size/(81*81*3))), dtype='uint8'))
    return data, labeled
    # save_tfl_data_as_binary(negative, labeled)


