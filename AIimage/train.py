# -*-coding:UTF-8-*-

"""训练DCGAN"""

import glob
import numpy as np
from scipy.misc import imread
import tensorflow as tf

from network import *

def train():
    data=[]
    for image in glob.glob("images/*"):
        image_data=imread(image)
        data.append(image_data)
    input_data=np.array(data)
    input_data=(input_data.astype(np.float32)-127.5)/127.5

    g=generate_model()
    d=discriminator_model()

    d_on_g=generate_containing_discriminator(g,d)

    g_optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE,beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)

    g.compile(loss="binary_crossentropy",optimizer=g_optimizer)
    d.compile(loss="binary_crossentropy",optimizer=d_optimizer)

    d.trainable=True
    d_on_g.compile(loss="binary_crossentropy",optimizer=g_optimizer)

    for epoch in range(EPOCHS):
        for index in range(int(input_data.shape[0]/BATCH_SIZE)):
            input_batch=input_data[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            #噪声
            random_data=np.random.uniform(-1,1,size=(BATCH_SIZE,100))
            generated_images=g.predict(random_data,verbose=0)
            #按列把真实的样例数值和噪声进行拼接
            input_batch=np.concatenate((input_batch,generated_images))
            output_batch=[1]*BATCH_SIZE+[0]*BATCH_SIZE

            d_loss=d.train_on_batch(input_batch,output_batch)

            d.trainable=False
            g_loss=d_on_g.train_on_batch(random_data,[1]*BATCH_SIZE)

            d.trainable=True

            print("step %s generator loss: %s discriminator loss: %s"%(index,g_loss,d_loss))

            if epoch%10==9:
                g.save_weights("generator_weight",True)
                d.sav_weights("discriminator_weight",True)

if __name__=="__main__":
    train()


