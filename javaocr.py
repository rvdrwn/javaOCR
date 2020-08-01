import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
import cv2
from ocrload import imageprepare
from ocrtrain import create_placeholders, forward_propagation
from pre import preprocess

tf.compat.v1.disable_eager_execution()

def pred(result):
    a = result[0]
    print('------')
    if a<=9:
        print(a)
    elif a >= 10 and a <= 35:
        print(chr(a+87))
    else:
        print(chr(a+29))

if __name__ == "__main__":

    tf.random.set_seed(1)
    seed = 3

    m, n_H0, n_W0, n_C0 = 1, 32, 32, 1
    X, Y, keep_prob = create_placeholders(n_H0, n_W0, n_C0, 63)


    Z3 = forward_propagation(X, keep_prob)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    new_saver = tf.compat.v1.train.import_meta_graph('my-model.ckpt.meta')

    while True:
        frame = cv2.imread('image.png', 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = preprocess(frame)
		
        #cv2.imshow('draw', frame)
        cv2.imwrite('draw.jpg', frame)

        #tva = imageprepare(Image.fromarray(frame))
        tva = imageprepare('draw.jpg')
        image = np.array(tva)
        np.save('frame.npy', image)

        #cv2.imshow('love', image.reshape(n_H0,n_W0,1))

        #print(image.shape)
        image = image.reshape(1,n_H0,n_W0,1)
        prediction = tf.argmax(Z3, 1)

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))
            result = sess.run(prediction, feed_dict = {X:image, keep_prob:0.9 })
            pred(result)
		
        cv2.waitKey(1)
