# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cv2
#batch_norm层
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.examples.tutorials.mnist import input_data
number=0

class Build_Data:
    '''
    读取数据
    '''

    def __init__(self,label,input_dt=(28,28,1),data_path='./fashion'
                        ,log_dir='./log/001'):
        self.label=label
        self.size=input_dt
        self.data_path=data_path
        self.log_dir=log_dir
        self.mnist = input_data.read_data_sets(self.data_path, one_hot=False)

    def get_Data(self):
        #minmax=preprocessing.MinMaxScaler()
        images_array=self.mnist.train.images
        label_index=np.where(self.mnist.train.labels==self.label)[0]
        #images_array=minmax.fit_transform(images_array)
        images_array=np.reshape(images_array[label_index],[-1,28,28,1])
        return images_array

    def save_Img(self,next_x,image_save_src_path,show_size=(5,5),category=0):#=(10,10)
        print('show_size',show_size)
        #next_x, _ = self.mnist.train.next_batch(show_size[0]*show_size[1])
        #next_x=next_x[0:show_size[0]*show_size[1]]

        #plt.title('检索准确率:%.4f' % correct_rate)
        f, a = plt.subplots(show_size[0], show_size[1], figsize=(10, 10))
        plt.suptitle('当前类别:%d' % category)
        #f.suptitle()
        for i in range(show_size[0]):
            #print('i',i)
            for j in range(show_size[1]):
                #print('j',j)
                tmp_x=next_x[i * show_size[0] + j].reshape([28,28])
                a[i][j].imshow(tmp_x,cmap='gray')
                a[i][j].axis('off')
        global number
        img_path = os.path.join(image_save_src_path, str(number) + '.png')
        number += 1
        #plt.show()
        plt.savefig(img_path)

    # def save_Img(self,img_np,image_save_src_path):
    #     '''
    #     :param img_np:batch_size,28,28,1
    #     :return:
    #     '''
    #     img_len=img_np.shape[0]
    #     if img_len>=2:
    #         img_len=2
    #     else:
    #         pass
    #     global number
    #     for i in range(img_len):
    #         curr_np=img_np[i]
    #         #print(curr_np.shape)
    #         curr_np=curr_np.reshape([self.size[0],self.size[1]]).astype(np.uint8)
    #         curr_img=Image.fromarray(curr_np)
    #         img_path=os.path.join(image_save_src_path,str(number)+'.png')
    #         number+=1
    #         curr_img.save(img_path)


    def generate_Data(self,batch_size,data_len=100):#data_len=1024
        #noise_data=np.random.randn(batch_size,data_len)
        noise_data = np.random.uniform(-1, 1, size=(batch_size, data_len))
        #noise_data=np.random.normal(0,1,[batch_size,data_len])
        return noise_data


class Generate_Model:
    '''
    构造生成式模型
    '''
    def __init__(self,noise_data,batch_size,layer1=100,
                 layer2=256,layer3=128,layer4=64,layer5=32,layer6=1):
        self.noise_data=noise_data
        self.batch_size=batch_size
        self.layer1=layer1
        self.layer2=layer2
        self.layer3=layer3
        self.layer4=layer4
        self.layer5=layer5
        self.layer6=layer6
        self.input_len=noise_data.get_shape()[1]
    def get_Variable(self,name,shape,dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.1)):
        return tf.get_variable(name,shape,dtype,initializer)
    def get_Param(self,conv_size=5):
        w1=self.get_Variable('w1',[self.input_len,self.layer1])
        b1=self.get_Variable('b1',[self.layer1])
        w2=self.get_Variable('w2',[conv_size,conv_size,self.layer2,self.layer1])
        b2=self.get_Variable('b2',[self.layer2])
        w3=self.get_Variable('w3',[conv_size,conv_size,self.layer3,self.layer2])
        b3=self.get_Variable('b3',[self.layer3])
        w4=self.get_Variable('w4',[conv_size,conv_size,self.layer4,self.layer3])
        b4=self.get_Variable('b4',[self.layer4])
        w5=self.get_Variable('w5',[conv_size,conv_size,self.layer5,self.layer4])
        b5=self.get_Variable('b5',[self.layer5])
        w6=self.get_Variable('w6',[conv_size,conv_size,self.layer6,self.layer5])
        b6=self.get_Variable('b6',[self.layer6])
        return [(w1,b1),(w2,b2),(w3,b3),(w4,b4),(w5,b5),(w6,b6)]
    def get_Model(self):
        assert self.noise_data.get_shape()[1]==self.layer1
        (w1,b1),(w2,b2),(w3,b3),(w4,b4),(w5,b5),(w6,b6)=self.get_Param()
        with tf.variable_scope('G_layer1'):
            #nw=tf.matmul(self.noise_data,w1)+b1
            nw=tf.reshape(self.noise_data,[self.batch_size,1,1,self.layer1])
            nw = batch_norm(nw,  is_training=True)
        with tf.variable_scope('G_conv2'):
            nw=tf.nn.conv2d_transpose(nw,filter=w2,output_shape=[self.batch_size,2,2,self.layer2],strides=[1,2,2,1],padding='SAME')
            nw=tf.nn.bias_add(nw,b2)
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)#decay=0.9, updates_collections=None,
            #nw=tf.nn.sigmoid(nw)
        with tf.variable_scope('G_conv3'):
            nw=tf.nn.conv2d_transpose(nw,filter=w3,output_shape=[self.batch_size,4,4,self.layer3],strides=[1,2,2,1],padding='SAME')
            nw=tf.nn.bias_add(nw,b3)
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            #nw=tf.nn.sigmoid(nw)
        with tf.variable_scope('G_conv4'):
            nw=tf.nn.conv2d_transpose(nw,filter=w4,output_shape=[self.batch_size,7,7,self.layer4],strides=[1,2,2,1],padding='SAME')
            nw=tf.nn.bias_add(nw,b4)
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
        with tf.variable_scope('G_conv5'):
            nw=tf.nn.conv2d_transpose(nw,filter=w5,output_shape=[self.batch_size,14,14,self.layer5],strides=[1,2,2,1],padding='SAME')
            nw=tf.nn.bias_add(nw,b5)
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            #nw=tf.nn.sigmoid(nw)
        with tf.variable_scope('G_conv6'):
            nw=tf.nn.conv2d_transpose(nw,filter=w6,output_shape=[self.batch_size,28,28,self.layer6],strides=[1,2,2,1],padding='SAME')
            nw=tf.nn.bias_add(nw,b6)
            nw=tf.nn.tanh(nw)
            #nw=tf.nn.relu(nw)
            #nw = batch_norm(nw, decay=0.9, updates_collections=None, is_training=True)
            #nw=tf.nn.sigmoid(nw)



        return nw,[w1,b1,w2,b2,w3,b3,w4,b4,w5,b5,w6,b6]

class Discriminant_Model:
    '''
    构造判别式模型
    '''
    def __init__(self,x_true,x_generate,batch_size,layer1=1,layer2=16,layer3=32,layer4=64,layer5=128,layer6=1):
        self.x_true=x_true
        self.x_generate=x_generate
        self.batch_size=batch_size
        self.layer1=layer1
        self.layer2=layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
    def get_Variable(self, name, shape, dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1)):
        return tf.get_variable(name, shape, dtype, initializer)
    def get_Param(self,conv_size=5):
        e1=self.get_Variable('e1',[conv_size,conv_size,self.layer1,self.layer2])
        i1=self.get_Variable('i1',[self.layer2])
        e2=self.get_Variable('e2',[conv_size,conv_size,self.layer2,self.layer3])
        i2=self.get_Variable('i2',[self.layer3])
        e3=self.get_Variable('e3',[conv_size,conv_size,self.layer3,self.layer4])
        i3=self.get_Variable('i3',[self.layer4])
        e4=self.get_Variable('e4',[conv_size,conv_size,self.layer4,self.layer5])
        i4=self.get_Variable('i4',[self.layer5])
        e5=self.get_Variable('e5',[self.layer5,self.layer6])
        i5=self.get_Variable('i5',[self.layer6])
        return [(e1,i1),(e2,i2),(e3,i3),(e4,i4),(e5,i5)]
    def get_Model(self):
        [(e1, i1), (e2, i2), (e3, i3), (e4, i4),(e5,i5)]=self.get_Param()
        assert self.x_true.get_shape()[3]==1
        assert self.x_generate.get_shape()[3]==1
        x_input=tf.concat([self.x_true,self.x_generate],0)
        #x_input.shape==>(batch_size*2,28,28,1)
        with tf.variable_scope('D_conv1'):
            nw=tf.nn.conv2d(x_input,filter=e1,strides=[1,1,1,1],padding='SAME')
            nw=tf.nn.bias_add(nw,i1)
            nw=tf.nn.leaky_relu(nw)
            #nw = batch_norm(nw,  is_training=True)
            #nw=tf.nn.sigmoid(nw)
            #nw=tf.nn.max_pool(nw,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        with tf.variable_scope('D_conv2'):
            nw=tf.nn.conv2d(nw,filter=e2,strides=[1,2,2,1],padding='SAME')
            nw=tf.nn.bias_add(nw,i2)
            nw=tf.nn.leaky_relu(nw)
            nw = batch_norm(nw, is_training=True)
            #nw=tf.nn.max_pool(nw,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        with tf.variable_scope('D_conv3'):
            nw = tf.nn.conv2d(nw, filter=e3, strides=[1, 2, 2, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, i3)
            nw = tf.nn.leaky_relu(nw)
            nw = batch_norm(nw, is_training=True)
            #nw=tf.nn.sigmoid(nw)
            #nw = tf.nn.max_pool(nw, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('D_conv4'):
            nw = tf.nn.conv2d(nw, filter=e4, strides=[1, 2, 2, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, i4)
            nw = tf.nn.leaky_relu(nw)
            nw = batch_norm(nw,  is_training=True)
            #nw = tf.nn.max_pool(nw, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #batch_size,4,4,128
        with tf.variable_scope('D_out'):
            print(nw.get_shape)
            nw=tf.nn.avg_pool(nw,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
            nw=tf.reshape(nw,[self.batch_size*2,self.layer5])
            nw=tf.matmul(nw,e5)+i5
            #nw=tf.sigmoid(nw)
            nw=tf.nn.sigmoid(nw)
        nw_len=nw.get_shape()[0]//2

        nw_true=nw[:nw_len,:]
        nw_gene=nw[nw_len:,:]
        # nw_true=tf.slice(nw,[0,0],[nw_len,-1])
        # nw_gene=tf.slice(nw,[nw_len,0],[-1,-1])
        return nw_true,nw_gene,[e1,i1,e2,i2,e3,i3,e4,i4,e5,i5]

def binary_crossentropy(target, output, from_logits=False):
    if not from_logits:
        # transform back to logits
        _epsilon =1e-7
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)


def _main(training_epochs=40,batch_size=32,

          checkpoint_dir='./log/010',label=9,
          image_save_src_path='./images/'+str(9)):
    if os.path.exists(image_save_src_path):
        pass
    else:
        os.mkdir(image_save_src_path)
    BD=Build_Data(label=label)
    #(200,64,64,1)
    x_true=BD.get_Data()
    #构建网络占位符
    X_G=tf.placeholder(tf.float32,[batch_size,100],name='X_G')
    X_D=tf.placeholder(tf.float32,[batch_size,28,28,1],name='X_D')
    global_step=tf.Variable(0,name='global_step',trainable=False)
    #构建生成模型
    ge_M=Generate_Model(X_G,batch_size)
    G_out,G_params=ge_M.get_Model()

    #构建判别模型
    di_M=Discriminant_Model(X_D,G_out,batch_size)
    D_true,D_gene,D_params=di_M.get_Model()

    #损失函数
    t_vars = tf.trainable_variables()
    G_vars=[var for var in t_vars if 'w' in var.name or 'b' in var.name or 'G_conv' in var.name]
    D_vars=[var for var in t_vars if 'e' in var.name or 'i' in var.name or 'D_conv' in var.name]
    g_loss1=tf.reduce_sum(tf.pow(D_gene-np.ones((batch_size,1),dtype=np.float32),2))
    #g_loss2=tf.reduce_sum(tf.pow(D_true-np.zeros((batch_size,1),dtype=np.float32),2))
    g_loss=(g_loss1)
    d_loss1=tf.reduce_sum(tf.pow(D_true-np.ones((batch_size,1),dtype=np.float32),2))
    d_loss2=tf.reduce_sum(tf.pow(D_gene-np.zeros((batch_size,1),dtype=np.float32),2))
    #d_loss=tf.reduce_mean(-tf.log(D_true))+tf.reduce_mean(-tf.log(1-D_gene))
    d_loss=(d_loss1+d_loss2)/2
    #d_loss=tf.reduce_mean(-(tf.log(D_true)+tf.log(1-D_gene)))
    #g_loss=tf.reduce_mean(-tf.log(D_gene))
    #定义优化函数
    g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0001,beta1=0.5)  #0.0001
    #g_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
    d_optimizer=tf.train.AdamOptimizer(learning_rate=0.00001,beta1=0.5)#0.000006
    #d_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #梯度更新，指定参数列表
    g_train=g_optimizer.minimize(g_loss,var_list=G_vars)
    d_train=d_optimizer.minimize(d_loss,var_list=D_vars)
    #模型保存方法
    saver=tf.train.Saver(max_to_keep=2)
    #开始训练模型
    with tf.Session() as sess:
        #变量初始化
        sess.run(tf.global_variables_initializer())
        ckpt=None
        if True:
            #加载模型并继续训练
            ckpt=tf.train.latest_checkpoint(checkpoint_dir)
            if ckpt:
                print('load model---')
                saver.restore(sess,ckpt)
        #每个epoch的数据长度
        data_len=x_true.shape[0]
        #批次步长
        steps=data_len//batch_size
        for epoch in range(sess.run(global_step),training_epochs):
            #获取随机索引
            index=np.random.permutation(data_len)
            g_loss_=0
            d_loss_=0
            d_time=0
            for batch in range(steps):
                batch_D=x_true[index[batch*batch_size:(batch+1)*batch_size]]
                batch_G=BD.generate_Data(batch_size=batch_size)
                sess.run(g_train,feed_dict={X_G:batch_G,X_D:batch_D})
                sess.run(d_train, feed_dict={X_G: batch_G, X_D: batch_D})
                g_loss_+=sess.run(g_loss,feed_dict={X_G:batch_G,X_D:batch_D})
                d_loss_+=sess.run(d_loss, feed_dict={X_G: batch_G, X_D: batch_D})
                # if g_loss_  <= 200:
                #     sess.run(d_train, feed_dict={X_G: batch_G, X_D: batch_D})
                #     d_time+=1
                #     d_loss_ += sess.run(d_loss, feed_dict={X_G: batch_G, X_D: batch_D})
            try:
                print('g_loss:%.4f,d_loss:%.4f'%(g_loss_/steps,d_loss_/steps))
            except BaseException:
                print('g_loss:{}'.format(g_loss_ / steps))
            show_batch=BD.generate_Data(batch_size=batch_size)
            #show_batch=batch_G
            generate_value=sess.run(G_out,feed_dict={X_G:show_batch})
            BD.save_Img(generate_value,image_save_src_path=image_save_src_path)
            sess.run(tf.assign(global_step,epoch+1))
            saver.save(sess,os.path.join(checkpoint_dir,'model'),global_step=global_step)

def predict(checkpoint_dir='./log/010',batch_size=32,
            save_images=10000,image_save_src_path='./extend_mnist/'+str(9)):
    if os.path.exists(image_save_src_path):
        pass
    else:
        os.mkdir(image_save_src_path)
    BD2=Build_Data(label=0)
    # 构建网络占位符
    X_G = tf.placeholder(tf.float32, [batch_size, 100], name='X_G')
    X_D = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='X_D')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # 构建生成模型
    ge_M = Generate_Model(X_G, batch_size)
    G_out, G_params = ge_M.get_Model()
    # 模型保存方法
    saver = tf.train.Saver(max_to_keep=2)
    flag=0
    # 开始训练模型
    with tf.Session() as sess:
        # 变量初始化
        sess.run(tf.global_variables_initializer())
        ckpt = None
        if True:
            # 加载模型并继续训练
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if ckpt:
                print('load model---')
                saver.restore(sess, ckpt)
        while flag<=save_images:
            save_batch = BD2.generate_Data(batch_size=batch_size)
            # show_batch=batch_G
            generate_value = sess.run(G_out, feed_dict={X_G: save_batch})
            for i in range(batch_size):
                curr_img=np.reshape(np.array(generate_value[i]),[28,28])
                #curr_img=Image.fromarray(curr_img)
                plt.figure(figsize=(28,28))
                plt.axis('off')
                plt.imshow(curr_img,cmap='gray')
                save_path=os.path.join(image_save_src_path,str(flag))
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig(save_path,transparent=True, pad_inches = 0)
                plt.close()
                #curr_img.save(save_path)
                #cv2.imwrite(save_path,curr_img)
                flag+=1

if __name__=='__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    #_main()
    predict()
