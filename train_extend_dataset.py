# _*_ coding:utf-8 _*_
#================================================
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
#=================gpu动态占用=====================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder
from keras import layers,models,backend,callbacks
from keras.optimizers import Adam
import utils.tools as utils

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

class FashionMNIST():

    def __init__(self,data_path='./fashion',batch_size=16,input_dt=(28,28,1),
                 classes=10,log_dir='./log/011/',feature_size=2048,
                 extend_path='extend_mnist',metrics_save_path='./result'):
        self.input_dt=input_dt
        self.classes=classes
        self.data_path=data_path
        self.batch_size=batch_size
        self.log_dir=log_dir
        self.extend_dir=extend_path
        self.save_path=metrics_save_path
        if os.path.exists(self.save_path):
            pass
        else:
            os.mkdir(self.save_path)
        self.start = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.feature_size=feature_size
        #self.x=tf.placeholder(tf.float32,shape=[None,self.input_dt])
        self.x=layers.Input(shape=[self.input_dt[0]*self.input_dt[1]])
        #self.y=tf.placeholder(tf.float32,shape=[None,self.output_dt])
        self.mnist = input_data.read_data_sets(self.data_path, one_hot=True)
        train_images = self.mnist.train.images
        train_labels = self.mnist.train.labels
        self.test_images = self.mnist.test.images
        self.test_labels = self.mnist.test.labels
        category_images = []
        category_labels = []
        for i in range(self.classes):
            classes_dir = os.path.join(self.extend_dir, str(i))
            classes_list = os.listdir(classes_dir)
            classes_list_len = len(classes_list)
            for j in range(classes_list_len):
                each_img_path = os.path.join(classes_dir, classes_list[j])
                each_img=Image.open(each_img_path)
                each_img = np.array(each_img).reshape([784])
                each_img=(each_img-each_img.min())*(1/(each_img.max()-each_img.min()))
                #each_img = each_img.resize((28, 28))
                #each_img = each_img.convert('L')
                #each_img.save(each_img_path)
                # if j>=10:
                #     break
                category_images.append(each_img)
                category_labels.append(i)
        category_images = np.array(category_images)
        category_labels = np.array(category_labels).reshape([-1, 1])
        category_labels = OneHotEncoder(sparse=True).fit_transform(category_labels).toarray()
        #print(train_images[0])
        #print(category_images[0])
        #exit()
        self.train_images = np.concatenate([train_images, category_images], axis=0)
        self.train_labels = np.concatenate([train_labels, category_labels], axis=0)
        print('训练集：images={},labels={}'.format(self.train_images.shape, self.train_labels.shape))
        print('测试集：images={},labels={}'.format(self.test_images.shape, self.test_labels.shape))

    def generate_Data(self):


        while 1:
            samples, _ = self.get_Examples()
            batch_num = samples // self.batch_size
            index = np.random.permutation(samples)
            for i in range(batch_num):
                next_x, next_y = self.train_images[index[i * self.batch_size:(i + 1) * self.batch_size]] \
                    , self.train_labels[index[i * self.batch_size:(i + 1) * self.batch_size]]
                yield next_x, next_y

    def generate_test_Data(self):
        #mnist = input_data.read_data_sets(self.data_path, one_hot=True)
        while 1:
            next_x, next_y = self.mnist.test.next_batch(self.batch_size)
            yield next_x, next_y

    def show_Data(self,next_x,show_size=(30,30)):
        #next_x=self.train_images[np.where(np.argmax(self.train_labels)==0)[0]]
        #next_x, _ = self.mnist.train.next_batch(show_size[0]*show_size[1])
        next_x=next_x[0:show_size[0]*show_size[1]]
        f, a = plt.subplots(show_size[0], show_size[1], figsize=(10, 10))
        for i in range(show_size[0]):
            for j in range(show_size[1]):
                tmp_x=next_x[i * show_size[0] + j].reshape([28,28])
                a[i][j].imshow(tmp_x,cmap='gray')
                a[i][j].axis('off')
        plt.show()

    def identity_block(self,input_tensor, kernel_size, filters, stage, block,bn_axis=3):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def conv_block(self,input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2),
                   bn_axis=3):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def reshape_Tensor(self,tensor):
        return backend.reshape(tensor,shape=[-1,28,28,1])

    def res_Net50(self,input_tensor,bn_axis=int(3)):
        x=layers.Lambda(self.reshape_Tensor)(input_tensor)
        #x=backend.reshape(input_tensor,shape=[-1,28,28,1])
        #self.x=layers.Input(tensor=x)
        x=layers.BatchNormalization(name='bn_conv0')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
        x = layers.Conv2D(64, (3, 3),
                          strides=(1, 1),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(x)
        x = layers.BatchNormalization( name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(1, 1))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a',strides=(1,1))
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(self.classes, activation='softmax', name='fc10')(x)
        model=models.Model(self.x,x,name='resnet50')
        return model

    def get_Examples(self):
        return self.train_labels.shape[0],self.test_images.shape[0]
        #return 1000,self.test_images.shape[0]

    def model_Train(self):

        train_exam,test_exam=self.get_Examples()

        model=self.res_Net50(self.x)
        # 指定回调函数
        logging = callbacks.TensorBoard(log_dir=self.log_dir)
        checkpoint = callbacks.ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_best_only=True, mode='min',
                                     save_weights_only=True, period=1)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        # 指定训练方式
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        if os.path.exists('./' + self.log_dir +'/'+ 'train_weights.h5'):
            model.load_weights('./' + self.log_dir +'/'+ 'train_weights.h5')
        #model.load_weights('/home/dream/PythonProjects/GAN_Fashion_MNIST/log/011/ep007-loss0.167-val_loss0.217.h5')
        model.fit_generator(self.generate_Data(),
                            steps_per_epoch=max(1,train_exam//self.batch_size),
                            validation_data=self.generate_test_Data(),
                            validation_steps=max(1,test_exam//self.batch_size),
                            epochs=15,
                            initial_epoch=10,
                            callbacks=[logging,checkpoint,reduce_lr,early_stopping])
        model.save_weights(self.log_dir+'/'+'train_weights.h5')
        model.save(self.log_dir+'/'+'train_models.h5')

    def calu_Accuracy(self):
        model = self.res_Net50(self.x)
        if os.path.exists(self.log_dir+'/'+'train_weights.h5'):
            model.load_weights(self.log_dir+'/'+'train_weights.h5')
        else:
            raise RuntimeError('load weights Error!')
        test_images=self.mnist.test.images
        test_labels=self.mnist.test.labels
        print("正在预测测试集样本")
        result=model.predict(test_images)
        with open('temp','w+')as file:
            for i in range(len(result)):
                file.write(str(result[i])+'\n')

        equal=np.equal(np.argmax(test_labels,axis=1),np.argmax(result,axis=1))
        equal=np.where(equal==True,1,0)
        #print(np.cast(equal,np.float32))
        correct_rate=np.mean(equal)
        print("测试集合准确率为：{}".format(correct_rate))

        fw_perf = open(self.save_path + '/index2.txt', 'w')
        fw_perf.write('acc' + ',' + 'precision' + ',' + 'npv' + ',' +
                      'sensitivity' + ',' + 'specificity' + ',' + 'mcc' + ',' +
                      'ppv' + ',' + 'auc' + ',' + 'pr' + '\n')
        auc_ = roc_auc_score(test_labels, result)
        pr = average_precision_score(test_labels, result)
        y_class = utils.categorical_probas_to_classes(result)
        true_y = utils.categorical_probas_to_classes(test_labels)
        acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(
            len(y_class), y_class, true_y)
        print("======================")
        print("======================")
        print(
            '\tacc=%0.4f,pre=%0.4f,npv=%0.4f,sn=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f'
            % (acc, precision, npv, sensitivity, specificity, mcc, f1))
        print('\tauc=%0.4f,pr=%0.4f' % (auc_, pr))

        fw_perf.write(
            str(acc) + ',' + str(precision) + ',' + str(npv) + ',' +
            str(sensitivity) + ',' + str(specificity) + ',' + str(mcc) +
            ',' + str(f1) + ',' + str(auc_) + ',' + str(pr) + '\n')
        end = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print('start: %s' % self.start)
        print('end: %s' % end)
        fw_perf.write('start: %s' % self.start+'\n')
        fw_perf.write('end: %s' % end+'\n')

if __name__=='__main__':
    fmn=FashionMNIST()
    fmn.show_Data(fmn.mnist.train.images)
    fmn.model_Train()
    fmn.calu_Accuracy()




