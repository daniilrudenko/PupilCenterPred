from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from ReadCSV import ReadData
from sklearn import metrics
import math

class MyModel:



#При создании объекта данного класса будет либо загружаться обученная модель,
#либо инициализация для обучения, в зависимости от того, есть аргумент конструктора.
#Начало общее - получение двух фреймов с данными обучения и теста.



    def __init__(self,data=None):
     #   self.data_train = ReadData(r'train_data/').split_data()
        self.data_test = ReadData(r'test_data/').split_data()
     #   self.path_toTrain = r'train_data/'
        self.path_toTest = r'test_data/'
        if data==None:
            self.model = models.Sequential()
            self.model = models.Sequential()
            self.model.add(layers.Conv2D(24, (8, 8), activation='relu', input_shape=(200, 200, 1),padding='same'),strides=(2,2))
            self.model.add(layers.Conv2D(36, (6, 6), activation='relu', padding='same'))
            self.model.add(layers.Conv2D(52, (5, 5), activation='relu', padding='same'),strides=(2,2))
            self.model.add(layers.Conv2D(80, (3, 3), activation='relu', padding='same'),strides=(2,2))
            self.model.add(layers.Conv2D(124, (3, 3), activation='relu', padding='same'),strides=(2,2))
            self.model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'), strides=(2, 2))
            self.model.add(layers.Conv2D(500, (3, 3), activation='relu', padding='same'), strides=(2, 2))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(2))
            self.model.summary()
        else:
            self.model = models.load_model(data)



#get_batch - функция генератор. В бесконечном цикле выдаёт выборку изображений
#и соответствующих позиций.


    def get_batch(self,n=1, path='', df=None, state=False):
        batch_images = np.zeros((n, 200, 200, 1), dtype=np.float32)
        batch_pos = np.zeros((n, 2))
        while True:
            for count in range(n):

                idx = np.random.randint(len(df))
                image = cv.imread(path + df['name']._get_value(idx), 0)
                image = image.reshape(200, 200, 1)
                image = image / 255.0
                batch_images[count] = image
                batch_pos[count] = df['pcx']._get_value(idx), df['pcy']._get_value(idx)
                if state:
                    print(path + df['name']._get_value(idx))
            yield batch_images, batch_pos



    def make_plot(self,his_first, his_sec, name, obj = None):
        plt.plot(obj.history[his_first], label=name + ' (training data)')
        plt.plot(obj.history[his_sec], label=name + ' (validation data)')
        plt.title(name)
        plt.ylabel(name + ' value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()




    def train(self,model_save_path):
        opt = optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizers=opt,
                      loss='mse',
                      metrics=['mae'])

        history = self.model.fit(self.get_batch(25, 'train_data/', self.data_train), epochs=1000, steps_per_epoch=224,
                            validation_data=self.get_batch(20, 'train_data/', self.data_train), validation_steps=28)

        self.model.save(model_save_path)
        self.make_plot('loss','val_loss','MSE',history)
        self.make_plot('mae','val_mae','MAE',history)



    def test_(self):
        print('\nTest:')
        res = self.model.evaluate(self.get_batch(25, self.path_toTest, self.data_test), steps=16)
        print(res)
        print('\n')

    def predict_(self):
        pred = self.model.predict(self.get_batch(4, self.path_toTrain, self.data_train, True), steps=1)
        print(pred)

    def get_model(self):
        return self.model


# Вычисление RMSE по отдельной координате


    def calc(self):
        mean = np.zeros((2, 1))
        rmse = np.zeros((400, 1))
        pos_x = np.zeros((1, 1))
        pos_x_pr = np.zeros((1, 1))
        img_tensor = np.zeros((1, 200, 200, 1), dtype=np.float32)
        for count in range(400):
            pos_x[0] = self.data_test['pcx']._get_value(count)
            image = cv.imread(self.path_toTest + self.data_test['name']._get_value(count), 0)
            image = image.reshape(200, 200, 1)
            image = image / 255.0
            img_tensor[0] = image
            predicted_data = self.model.predict(img_tensor)
            pos_x_pr[0] = predicted_data[0][0]
            mse = metrics.mean_squared_error(pos_x[0], pos_x_pr)
            rms = math.sqrt(mse)

            rmse[count] = rms
            print(self.data_test['name']._get_value(count))
            print(rms)
        mean[0] = np.mean(rmse)
        print('\n')
        print(mean)
        plt.plot(rmse, label='RMSE по координате X')
        plt.axhline(mean[0],color = 'r',label='Cреднее значение')
        plt.title('RMSE X')
        plt.ylabel('RMSE')
        plt.xlabel('Номер фрейма')
        plt.legend(loc="upper left")
        plt.show()

    def calc_pix_err(self):
        mean = np.zeros((2, 1))
        error = np.zeros((400, 1))
        pos_x = np.zeros((1, 1))
        pos_x_pr = np.zeros((1, 1))
        img_tensor = np.zeros((1, 200, 200, 1), dtype=np.float32)
        for count in range(400):
            pos_x[0] = self.data_test['pcx']._get_value(count)
            image = cv.imread(self.path_toTest + self.data_test['name']._get_value(count), 0)
            image = image.reshape(200, 200, 1)
            image = image / 255.0
            img_tensor[0] = image
            predicted_data = self.model.predict(img_tensor)
            pos_x_pr[0] = predicted_data[0][0]
            error[count] = pos_x[0] - pos_x_pr[0]
            print(pos_x[0])
            print(pos_x_pr[0])
            print(np.absolute(error[count]))
            print(self.data_test['name']._get_value(count))
        a = np.absolute(error)
        mean[0] = np.mean(a)
        print('\n')
        print(mean)
        plt.plot(a, label='Pixel error по координате X')
        plt.axhline(mean[0], color='r', label='Cреднее значение')
        plt.title('Pixel error X')
        plt.ylabel('Pixel error')
        plt.xlabel('Номер фрейма')
        plt.legend(loc="upper left")
        plt.show()



    def make_pred_img_vis(self, type = 0,str=''):
        if type == 0:
            df = self.data_test
            path = self.path_toTest
        else:
            df = self.data_train
            path = self.path_toTrain
        if str=='':
            idx = np.random.randint(len(df))
            val = df['name']._get_value(idx)
        else:
            val = str
        image = cv.imread(path + val, 0)
        imgForPred = image
        image = image.reshape(200, 200, 1)
        image = image / 255.0
        batch_images = np.zeros((1, 200, 200, 1), dtype=np.float32)
        batch_images[0] = image
        r = self.model.predict(batch_images)
        x1, y1 = r[0]
        print(val)
        print('rounded: %d %d' % (round(x1),round(y1)))
        print('not rounded: %f %f' % (r[0][0],r[0][1]))
        img_col = cv.cvtColor(imgForPred,cv.COLOR_GRAY2RGB)
        cv.line(img_col, (round(x1), round(y1)), (round(x1), round(y1)), (0, 255, 0), 3)
        cv.imshow(path + val, img_col)
        cv.waitKey(0)

    def make_pred2(self,str=''):
        im = cv.imread(str,0)
        image = cv.resize(im, dsize=(200, 200), interpolation=cv.INTER_CUBIC)
        imgForPred = image
        image = image.reshape(200, 200, 1)
        image = image / 255.0
        batch_images = np.zeros((1, 200, 200, 1), dtype=np.float32)
        batch_images[0] = image
        r = self.model.predict(batch_images)
        x1, y1 = r[0]
        print(str)
        print('rounded: %d %d' % (round(x1), round(y1)))
        print('not rounded: %f %f' % (r[0][0], r[0][1]))
        img_col = cv.cvtColor(imgForPred, cv.COLOR_GRAY2RGB)
        cv.line(img_col, (round(x1), round(y1)), (round(x1), round(y1)), (0, 255, 0), 3)
        cv.imshow(str, img_col)
        cv.waitKey(0)
