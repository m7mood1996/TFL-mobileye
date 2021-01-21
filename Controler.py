
from TFL_Man import TFL_Man
import json

class Controler():

    def __init__(self):
        self.tfl_man = TFL_Man()
        self.pkl_path, self.frames_path = self.get_data_pathes()

    def get_data_pathes(self):

        with open('pls.json') as pls:
            pls_data = json.load(pls)

        pkl_path = pls_data['pkl']
        frames_path = [pls_data['frame' + str(i)] for i in range(1, len(pls_data))]
        return pkl_path, frames_path

        # run controller's managers
    def run(self):
        self.tfl_man.run(self.pkl_path, self.frames_path)



if __name__ == '__main__':
    controler = Controler()
    controler.run()