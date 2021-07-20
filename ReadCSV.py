import fnmatch
import os
import pandas as pd

"""
Класс по работе с CSV файлом. 
Данные в файле распределены так - "название изображения" и координаты 
X Y для левого глаза, и правого. 
Работа данного класса заточена на то, что бы создать фрейм, где сперва будет
один элемент данных, например левый глаз, а затем правый.

"""


class ReadData:
    def __init__(self, path=None):
        self.path = path
        self.data = None


        if self.path:
            self.data = self.read_data()

    def read_data(self):
        files = os.listdir(self.path)
        files = fnmatch.filter(files, '*csv*')
        self.data = [pd.read_csv(os.path.join(self.path, n)) for n in files]
        self.data = pd.concat(self.data, axis=0).reset_index(drop=True)
        return self.data

    def split_data(self):
        df = []
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            name = row['name'].split('.')
            df.append({'name': f'{name[0]}.L.{name[1]}', 'pcx': row.pcx_l, 'pcy': row.pcy_l})
            df.append({'name': f'{name[0]}.R.{name[1]}', 'pcx': row.pcx_r, 'pcy': row.pcy_r})
        return pd.DataFrame(df)
