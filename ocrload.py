import os
import numpy as np
from PIL import Image, ImageFilter
import cv2

parent_path = './dataset' #menggunakan folder 'dataset' sebagai parent path
all_directories = os.listdir(parent_path+ '/')

#initialisasi in/out training & test
in_train = []
out_train = []
in_test = []
out_test = []

count = 0 #initialisasi count untuk loading image

def imageprepare(argv):    
    im = Image.open(argv).convert('L') #membuka file image dan mengkonversi image menjadi 8 bit hitam putih (mode 'L')
    lebar = float(im.size[0]) #diambil value pertama pada vector image size (x)
    tinggi = float(im.size[1]) #diambil value kedua pada vector image size (y)
    newImage = Image.new('L', (32,32), (255)) #membuat image 8 bit hitam putih baru dengan size 32*32
    
    if lebar > tinggi:
        tinggi_baru = int(round((30/lebar*tinggi),0)) #membuat tinggi baru (nheight) 
        
		# preprocess
        img = im.resize((30,tinggi_baru), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        offsety = int(round(((32 - tinggi_baru)/2),0)) #menghitung offset image agar centered
        newImage.paste(img, (4, offsety)) #paste image pada canvas
    else:
        lebar_baru = int(round((30/tinggi*lebar),0)) #membuat lebar baru
        
		# preprocess
        img = im.resize((lebar_baru,30), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        offsetx = int(round(((32 - lebar_baru)/2),0))
        newImage.paste(img, (offsetx, 4)) 
	
    #newImage.save("test.png")
    pix = list(newImage.getdata()) #membuat list dengan isi nilai pixel image
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    pixs = [(255-x)/255 for x in pix] #melakukan normalisasi nilai pixel menjadi 0 untuk putih dan 1 untuk hitam
    return pixs

def load_data():
    global count
	
	#melakukan scanning pada folder-folder yang tertulis
    for i in ['BA', 'CA', 'DA', 'DHA', 'GA', 'HA', 'JA', 'KA', 'LA', 'MA', 'NA', 'NGA', 'NYA', 'PA', 'RA', 'SA', 'TA', 'THA', 'WA', 'YA']:
        for z in all_directories:
            output_vector = np.zeros((63,1))
            path = parent_path + '/' + z

            if z.startswith(str(i)) and os.path.isdir(path):
                print(path)

                total_files = len(os.listdir(path))
                training_files = int(90/100 * total_files)
                test_files = total_files - training_files

                lim = 0
                for file_name in os.listdir(path):
                    pixs = imageprepare(path + '/' + file_name)
                    input_image = np.array(pixs)
                    input_image = input_image.reshape(32,32,1) #mengubah array input image menjadi vector 32x32
                    if lim == 0: #menunjukkan image pertama setiap folder
                        cv2.imshow('abcd', input_image)
                        cv2.waitKey(1)
                    if lim < training_files:
                        in_train.append(input_image) #memasukkan nilai vector input image kedalam in_train
                        output_vector[count] = 1
                        out_train.append(output_vector) #memasukkan nilai output_vector pada out_train
                    elif lim < training_files + test_files:
                        in_test.append(input_image)
                        output_vector[count] = 1
                        out_test.append(output_vector)
                    else:
                        break
                    lim += 1
        count += 1

    return in_train, out_train, in_test, out_test

if __name__ == "__main__":
    in_train, out_train, in_test, out_test = load_data()
    np.save('in_train.npy', in_train)
    np.save('out_train.npy', out_train)
    np.save('in_test.npy', in_test)
    np.save('out_test.npy', out_test)
