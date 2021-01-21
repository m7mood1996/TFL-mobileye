
import json
from TFL_Detection import tfl_basic_detiction
from TFL_validate import tfl_validation
from TFL_Destince import tfl_destince
import numpy as np
import matplotlib.pyplot as plt


class TFL_Man():

    def __init__(self):
        self.model = self.get_model()
        self.prev = []
        self.curr = []


    def get_model(self):
        return tfl_validation.get_tfl_detaction_model()

    def validate_images(self, images, labeled, model):
        validated = []
        i = 0
        for x, y, image in images:
            if tfl_validation.validate_tfl(model, image):
                validated.append([x, y, labeled[i]])
            i += 1

        return validated


    def run(self, pkl_path, frames_path):
        i = 0
        w = 10
        h = 10
        fig = plt.figure(figsize=(8, 8))
        columns = 1
        rows = 3
        for frame in frames_path:
            # part 1
            images, labeled = tfl_basic_detiction.main_tfl_basic_detiction(frame)
            f = fig.add_subplot(rows, columns, 1)
            f.set_title("First Part : TFL detection")
            f.plot([x[0] for x in images], [x[1] for x in images], 'b+')
            plt.imshow(plt.imread(frame))

            # part 2
            self.curr = self.validate_images(images, labeled, self.model)
            f = fig.add_subplot(rows, columns, 2)
            f.set_title("Second Part : Validate TFL")
            f.plot([x[0] for x in self.curr], [x[1] for x in self.curr], 'b+')
            plt.imshow(plt.imread(frame))

            # part 3
            if not i == 0:
                tfl_destince.get_dest(pkl_path, frames_path[i-1], frame, i + 23,i + 24, self.prev, self.curr, fig)

            self.prev = self.curr
            i += 1
            plt.show()
            fig = plt.figure(figsize=(8, 8))
