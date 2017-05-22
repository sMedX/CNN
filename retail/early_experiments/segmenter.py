#! /usr/bin/python3

import numpy as np
import unittest

class Segmenter:
    def __init__(self, model, patch_size, num_classes):
        self.model = model
        self.patch_size = patch_size
        self.num_classes = num_classes

    def segment_image(self, image):
        width = image.shape[0]
        height = image.shape[1]
        new_width_in_patches = width // self.patch_size
        new_width = new_width_in_patches * self.patch_size
        new_height_in_patches = height // self.patch_size
        new_height = new_height_in_patches * self.patch_size
        skip_width = (width - new_width) // 2
        skip_height = (height - new_height) // 2
        cropped_image = image[skip_width:skip_width+new_width, skip_height:skip_height+new_height, :]

        probs = np.zeros((new_width//self.patch_size, new_height//self.patch_size, self.num_classes))

        for i in range(new_width_in_patches):
            for j in range(new_height_in_patches):
                X = cropped_image[i*self.patch_size:(i+1)*self.patch_size,
                                  j*self.patch_size:(j+1)*self.patch_size]
                X = np.expand_dims(X, axis = 0)
                probs[i, j, :] = self.model.predict(X)

        return cropped_image, probs

    def segment_and_color_image(self, image):
        image, probs = self.segment_image(image)

        width_in_patches = image.shape[0] // self.patch_size
        height_in_patches = image.shape[1] // self.patch_size

        palette = [
            0x410D31,
            0x413C42,
            0x627971,
            0x671430,
            0x708258,
            0x7C6E4B,
            0x803530,
            0x8A9C82,
            0xA87943,
            0xCAAD51,
            0xCDB6A8,
            0xD4CB66,
        ]
        palette = [[x // 256 // 256, x // 256 % 256, x %256] for x in palette]
        palette = np.array(palette).astype(np.float32)

        image = image.astype(np.float32)

        blend = 0.7
        palette *= blend
        image *= 1. - blend

        for i in range(width_in_patches):
            for j in range(height_in_patches):
                x = i * self.patch_size
                y = j * self.patch_size

                index = np.argmax(probs[i, j])
                if index == 0: continue

                color = palette[index]

                image[x:x+self.patch_size, y:y+self.patch_size] += color


        return image

class ModelMock:
    num_classes = 2

    def predict(self, X):
        num_images = X.shape[0]
        output = np.zeros((num_images, num_images))
        for i in range(num_images):
            if X[i].mean() > 0:
                output[i, 1] = 1.
            else:
                output[i, 0] = 1.
        return output



class TestSegmenter(unittest.BaseTestSuite):
    def test_segment_image(self):
        pass
