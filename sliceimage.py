# Credit Karasu#0875 on Discord

from PIL import Image
import numpy as np


def main():

    n = 0
    r = 0

    image = "source_img.png"
    input_image = Image.open(image)

    # Assumes image resolution of 3840x2304 (1280x768 upscale 3 times)
    coordinates = [
        "0,0,1280,768",
        "1152,0,2432,768",
        "2304,0,3584,768",
        "3456,0,3840,768",
        "0,640,1280,1408",
        " 1152,640,2432,1408",
        "2304,640,3584,1408",
        "3456,640,3840,1408",
        "0,1280,1280,2048",
        " 1152,1280,2432,2048",
        "2304,1280,3584,2048",
        "3456,1280,3840,2048",
        "0,1920,1280,2304",
        " 1152,1920,2432,2304",
        "2304,1920,3584,2304",
        "3456,1920,3840,2304",
    ]

    for cord in coordinates:

        left, upper, right, lower = np.array([cord.split(",")], dtype=int).T
        xy_loc = np.concatenate([left, upper, right, lower])
        slices = input_image.crop((xy_loc))

        if n % 4 == 0:
            r += 1

        output_slices = "sliceimg{}-{}.png".format(r, n + 1)

        slices.save(output_slices)
        print("Slicing: {}".format(xy_loc))
        n += 1


if name == "main":
    main()
