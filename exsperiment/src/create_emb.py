import tensorflow as tf
import argparse
import random
import numpy as np
from PIL import Image
from skimage.util import view_as_blocks
import tensorflow_addons as tfa
from keras.applications.resnet import ResNet152
from keras_balanced_batch_generator import make_generator
import argparse
from datetime import timedelta
from time import perf_counter as pc


"""
Verifica di presenza della GPU sulla macchina utilizzata.
Setteggio di alcuni paremetri uitli per l'addestramento della 
triplet network. 
"""

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

seed=0
img_size = 96
batch_size = 32
epochs = 20
lr = 1e-5
wd = 1e-4
m = 0.1
emb_size = 512
distance ='L2'

"""
Settaggio dei semi random per la riproducibilità dei risultati.
"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

"""
Parser che permette di leggere gli argomenti
passati mediante linea di comando.
"""
def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fold", type=int,required=True, choices=[1,2,3,4,5], help="Fold")
    args = parser.parse_args()
    fold = str(args.fold)
    return fold
 
"""
Data una immagine di dimensione 460x700 la divide in patch
di dimensione 96x96 e le salva in un vettore. Il vettore finale
contiene 28 patches. 
"""
def create_patches(img):
    block_shape = np.array((img_size,img_size, 3))
    nblocks = np.array(img.shape) // block_shape  # integer division
    crop_r, crop_c, crop_ch = nblocks * block_shape
    cropped_img = img[:crop_r, :crop_c, :crop_ch]
    blocks = view_as_blocks(cropped_img, block_shape=(img_size,img_size,3))
    patches = []
    for x in blocks:
        for y in x:
            patches.append(y[0])
    return patches

"""
A partire dal file txt relativo al fold passato in input che 
contiene la ripartizione train-test ricava le features e le labels
del train. 
"""
def upload_train_set(f):
    path="dsfold%s.txt" %(f)
    x_train = []
    y_train = []

    mapping = {'b':0,'m':1}

    root_dir = '../BreaKHis_v1/histology_slides/breast'
    srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}
    
    path ="../src/dsfold"+str(f)+".txt"
    db = open(path)
    print("Training set creation....")

    for row in db.readlines():
        columns = row.split('|')
        imgname = columns[0]
        mag = columns[1]  # 40, 100, 200, or 400
        grp = columns[3].strip()  # train or test
        tumor = imgname.split('-')[0].split('_')[-1]
        srcfile = srcfiles[tumor]
        s = imgname.split('-')
        label = s[0].split("_")[1].lower()
        sub = s[0] + '_' + s[1] + '-' + s[2]
        srcfile = srcfile % (root_dir, sub, mag, imgname)
        image = Image.open(srcfile)
        x = np.asarray(image)
        patches = create_patches(x)
        if grp == 'train':
            x_train += patches
            y_train += len(patches)*[mapping[label]]
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    print("Number of patches in the training set:",len(x_train))
    print("Done !")
    return x_train,y_train

"""
Funzione per la creazione della triplet network.
"""
def create_triplet_net():
    triplet_net = tf.keras.models.Sequential()
    resnet152=ResNet152(include_top=False,
                   input_shape=(img_size,img_size,3),
                   pooling='max',classes=None,
                   weights='imagenet')

    triplet_net.add(resnet152)
    triplet_net.add(tf.keras.layers.Flatten())
    triplet_net.add(tf.keras.layers.Dense(1024,activation ='relu'))
    triplet_net.add(tf.keras.layers.Dense(emb_size, activation=None)) 
    triplet_net.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    opt = tf.keras.optimizers.legacy.Adam(learning_rate = lr, decay = wd)
    loss_fn = tfa.losses.TripletSemiHardLoss(margin=m, distance_metric = distance)
    triplet_net.compile(optimizer=opt,loss=loss_fn)
    return triplet_net

"""
Funzione che addestra la triplet network e salva i pesi finali.
"""
def train_net(triplet_net,x_train,y_train,f):
    steps_per_epoch = int( np.ceil(x_train.shape[0] / batch_size) )
    y_cat = tf.keras.utils.to_categorical (y_train)
    train_gen = make_generator(x_train, y_cat, batch_size=batch_size,
               categorical=False,
               seed=None)
    print("Training.....")
    start = pc()
    history = triplet_net.fit(train_gen,epochs=epochs,steps_per_epoch=steps_per_epoch)
    end = pc()-start
    training_time = timedelta(seconds=end)
    print("Training ended in:", str(training_time))
    path="../models/triplet_net_f%s.h5" %(f)
    triplet_net.save(path)
    print("Model saved!")
    return triplet_net

"""
Funzione che calcola gli embeddings delle patches 
di training e le salva in un array numpy.
"""
def save_embedding(triplet_net,f,x_train,y_train):
    print("Embeddings creation....")
    x_train_emb = triplet_net.predict(x_train,verbose=1)

    train_emb = "../embeddings/train/train_emb_f"+str(f)+".npz"

    np.savez_compressed(train_emb,x_train_emb,y_train)
    print("Embedding saved !")

"""
Main routine.
"""
def main():
    set_seed(seed)
    f = parse_argument()
    print("FOLD",f)
    x_train,y_train=upload_train_set(f)
    net=create_triplet_net()
    net=train_net(net,x_train,y_train,f)
    save_embedding(net,f,x_train,y_train)


if __name__=="__main__":
    main()
