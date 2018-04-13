# -*-coding:UTF-8-*-

"""用DCgan的生成器模型和训练得到的生成器参数来生成图片"""

import numpy as np
import tensorflow as tf

from PIL import Image
from network import *

def generate():
    g=generate_model()
    g.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE,beta_1=BETA_1))

    g.load_weights("generator_weight")
    random_data=np.random.uniform(-1,1,size=(BATCH_SIZE,100))
    images=g.predict(random_data,verbose=1)

    for i in range(BATCH_SIZE):
        images=images[i]*127.5+127.5
        Image.fromarray(images.astype(np.uint8)).save("image-%s.png"%i)

if __name__=="__main__":
    generate()





