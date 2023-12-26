import random 
import numpy as np 
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.applications.resnet import ResNet152
import tensorflow_addons as tfa
from PIL import Image
from skimage.util import view_as_blocks
import collections
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from torchmetrics.classification import Accuracy,Precision,F1Score,AUROC,Recall
from torchmetrics.classification import ConfusionMatrix
import torch
from minisom import MiniSom 
import matplotlib.gridspec as gridspec
import pickle

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
plt.rcParams['text.usetex'] = True


seed=0
img_size = 96
batch_size = 32
epochs = 20
lr = 1e-5
wd = 1e-4
m = 0.1
emb_size = 512
distance ='L2'
nfolds=5
n_neurons = 10
m_neurons = 10
dpi=1000
max_iter=100

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

def upload_data(f):
    path="../embeddings/train/train_emb_f%d.npz" %(f)
    data=np.load(path)
    x=data["arr_0"]
    y=data["arr_1"]

    print("Number of patches:",len(x))
    return x,y

"""
Data una immagine di dimensione 460x700 la divide in patch
di dimensione 96x96 e le salva in un vettore. Il vettore finale
contiene 28 patches. 
"""
def create_patches(path):
    x=Image.open(path)
    img=np.array(x)
    block_shape = np.array((img_size,img_size, 3))
    nblocks = np.array(img.shape) // block_shape  # integer division
    crop_r, crop_c, crop_ch = nblocks * block_shape
    cropped_img = img[:crop_r, :crop_c, :crop_ch]
    blocks = view_as_blocks(cropped_img, block_shape=(img_size,img_size,3))
    patches = []
    for x in blocks:
        for y in x:
            patches.append(y[0])
    return np.array(patches)

def upload_net(f):
    path="../models/triplet_net_f%d.h5" %(f)
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

    triplet_net.load_weights(path)

    return triplet_net

def save_som_plot(som,x,y,f):
    path="../plots/u_matrix%d.png" %(f)

    plt.figure(figsize=(10, 9))
    plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path,dpi=dpi)
    plt.clf()
    plt.close()


    label_names={
        0:"benign",
        1:"malignant"
    }


    labels_map = som.labels_map(x, [label_names[t] for t in y])


    path="../plots/sample_distf%d.png" %(f)
    fig = plt.figure()
    the_grid = gridspec.GridSpec(n_neurons, m_neurons, fig)
    for position in labels_map.keys():
        label_fracs = [labels_map[position][l] for l in label_names.values()]
        plt.subplot(the_grid[n_neurons-1-position[1],position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)


    #plt.legend(patches, label_names.values(), ncol=2)
    plt.savefig(path,dpi=dpi)
    plt.clf()
    plt.close()



"""
Utilizzo della SOM per il clustering.
"""
def apply_SOM(x,y,f):
    # Initialization and training
    som = MiniSom(n_neurons, m_neurons, x.shape[1], sigma=10, learning_rate=.5, 
              neighborhood_function='gaussian', random_seed=0, topology='rectangular')

    som.pca_weights_init(x)
    #som.train(x, 1000, verbose=True)  # random training

    q_error = []
    t_error = []


    for i in range(max_iter):
        rand_i = np.random.randint(len(x))
        som.update(x[rand_i], som.winner(x[rand_i]), i, max_iter)
        q_e=som.quantization_error(x)
        t_e=som.topographic_error(x)
        q_error.append(q_e)
        t_error.append(t_e)
        print("Iteration %d Quantization error: %f Topographic error %f" %(i,q_e,t_e))
    

    path="../plots/error_f%d.png" %(f)

    plt.figure()
    plt.plot(np.arange(max_iter), q_error, label='quantization error')
    plt.plot(np.arange(max_iter), t_error, label='topographic error')
    plt.ylabel('error')
    plt.xlabel('iteration index')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path,dpi=dpi)
    plt.clf()
    plt.close()

    winmap = som.labels_map(x,y)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]

    print("Number of cluster identified:",len(winmap))

    data=[]
    for c in winmap:
        aus=winmap[c]
        data.append((c,aus[0],aus[1]))

    df=pd.DataFrame(data,columns=["Unit position","Benign","Malignant"])

    path="../results/cluster_f%d.csv" %(f)
    df.to_csv(path,index=False)

    return som,winmap,default_class


def create_patches(path):
    img=np.array(Image.open(path))
    block_shape = np.array((img_size,img_size, 3))
    nblocks = np.array(img.shape) // block_shape  # integer division
    crop_r, crop_c, crop_ch = nblocks * block_shape
    cropped_img = img[:crop_r, :crop_c, :crop_ch]
    blocks = view_as_blocks(cropped_img, block_shape=(img_size,img_size,3))
    patches = []
    for x in blocks:
        for y in x:
            patches.append(y[0])
    return np.array(patches)

def save_prototype(som,x,f):
    path="../embeddings/paths_f%d" %(f)

    with open(path,"rb") as file:
        paths=pickle.load(file)
    
    distances=som._distance_from_weights(x).T

    indicies=np.argmin(distances,axis=1)
    

    fig_path="../plots/prototypes_f%d.png" %(f)

    k=0
    fig, ax = plt.subplots(10, 10)
    for i in range(n_neurons):
        for j in range(m_neurons):
            index=indicies[k]
            patch_info=paths[index]
            path=patch_info[0]
            patch_index=patch_info[1]
            images=create_patches(path)
            ax[i,j].imshow(images[patch_index])
            ax[i,j].axis("off")
            k+=1
    fig.savefig(fig_path,dpi=1000)
    plt.clf()
    plt.close()

    



"""
Clustering delle patches di test.
"""    
def assign_cluster(som,winmap,default_class,patches):
    result = []
    for d in patches:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

"""
Funzione che preso in input un vettore di label restituisce la moda.
Majority voting. 
"""
def assign_label(pred):
    counter=collections.Counter(pred)
    label = max(counter, key=counter.get)
    return label,counter


def frame_image(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    framed_img = Image.new('RGB', (b+ny+b, b+nx+b), (255, 0, 0)) # RGB color tuple
    framed_img = np.array(framed_img.getdata()).reshape(framed_img.size[0], framed_img.size[1], 3)
    framed_img[b:-b, b:-b] = img
    return framed_img

def plot_result(path,preds,pl,tl,f):
    patches=create_patches(path)
    indices = [i for i, x in enumerate(preds) if x ==1]
    mapping_labels={
        0:"Benign",
        1:"Malignant"
    }

    title="True label: %s - Predicted label: %s" %(mapping_labels[tl],mapping_labels[pl])

    k=0
    name=path.split("/")[-1].split(".")[0]
    fig_path="../plots/exp/%d/%s_exp.png" %(f,name)
    fig, ax = plt.subplots(4, 7)
    plt.suptitle(title)
    for i in range(4):
        for j in range(7):
            if k in indices:
                img=frame_image(patches[k],5)
            else:
                img=patches[k]
            ax[i,j].imshow(img)
            ax[i,j].axis("off")
            k+=1
    
    fig.savefig(fig_path,dpi=dpi)
    plt.clf()
    plt.close()
    

def predict_class(net,som,winmap,default_class,f):
    path="dsfold%d.txt" %(f)

    mapping = {'b':0,'m':1}
    y_test,y_pred=[],[]

    data=[]

    names=[]

    root_dir = '../BreaKHis_v1/histology_slides/breast'
    srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}
    
    db = open(path)
    
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
        if grp == "test":
            patches=create_patches(srcfile)
            y_test.append(mapping[label])
            emb_patches=net.predict(patches,verbose=0)
            patches_pred=assign_cluster(som,winmap,default_class,emb_patches)
            pred,counter=assign_label(patches_pred)
            """
            if(f==1):
                plot_result(srcfile,patches_pred,pred,mapping[label],f)
            """
            data.append((imgname,counter[0],counter[1]))
            y_pred.append(pred)
            names.append(imgname)
    
    df=pd.DataFrame(data,columns=["Img. name","Benign","Malignant"])
    path="../results/res_f%d.csv" %(f)
    df.to_csv(path,index=False)

    return y_test,y_pred,names

"""
Funzione per il calcolo della Patient Level Accuracy (PLA)
"""
def compute_pla(y_test,y_pred,paths,f):
    aus=[]
    for p in paths:
        x=p.split("-")
        aus.append(x[2])
    patients = list(dict.fromkeys(aus))
    p_i=[]
    for p in patients:
        indicies = [i for i, x in enumerate(paths) if p in x]
        y_true = [y_test[index] for index in indicies]
        y = [y_pred[index] for index in indicies]
        acc=accuracy_score(y_true,y)        
        p_i.append(acc)
        print("Patient Code: %s Patient score: %f" %(p,acc))
    
        
    pla=sum(p_i)/len(patients)

    data=list(zip(patients,p_i))

    path="../plots/patients_scores_f%d.png" %(f)
    df=pd.DataFrame(data,columns=["Patient","Score"])
    plt.figure()
    g=sns.barplot(data=df, x="Patient", y="Score")
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(path,dpi=dpi)
    plt.clf()
    return pla

"""
Funzione per il plot della matrice di confusione.
"""
def plot_cm(cm,f):
    path="../plots/cm%d.png" %(f)
    
    labels=["Benign","Malignant"]

    title="Confusion matrix - Fold %d" %(f)

    print(np.array(cm))

    plt.figure(figsize=(4,4))
    plt.title(title)
    ax=sns.heatmap(np.array(cm), annot=True,fmt='.2g',cmap='Blues',cbar=False,annot_kws={"size": 12})
    ax.set_xticklabels(labels,rotation=45,fontsize=12)
    ax.set_yticklabels(labels,rotation=45,fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.tight_layout()
    plt.savefig(path,dpi=dpi)
    plt.clf()
    plt.close()

"""
Funzione per il calcolo delle metriche di classificazione.
"""
def compute_metrics(y_true,y_pred,names,f):
    pla=compute_pla(y_pred,y_true,names,f)

    y_true=torch.tensor(y_true)
    y_pred=torch.tensor(y_pred)

    accuracy=Accuracy(task="binary",num_classes=2)
    acc=accuracy(y_true,y_pred).item()

    precision=Precision(task="binary",num_classes=2)
    prec=precision(y_true,y_pred).item()

    recall=Recall(task="binary",num_classes=2)
    rec=recall(y_true,y_pred).item()

    f1score=F1Score(task="binary",num_classes=2)
    f1=f1score(y_true,y_pred).item()

    confusion_matrix=ConfusionMatrix(task="binary",num_classes=2,normalize="true")
    cm=confusion_matrix(y_true,y_pred)
    plot_cm(cm,f)
    
    

    print("Accuracy:",acc)
    print("PLA:",pla)
    print("Precision",prec)
    print("Recall",rec)
    print("F1-Score:",f1)

    return [f,acc,pla,prec,rec,f1]


"""
Funzione per il salvataggio delle metriche di classificazione.
"""
def save_metrics(metrics):
    columns=["Fold","Accuracy","PLA","Precision","Recall","F1-Score"]
    data=pd.DataFrame(metrics,columns=columns)
    path="../results/metrics.csv"
    data.to_csv(path,index=False)

    avg_metrics=[]
    yaml_data={}

    acc=data['Accuracy'].values
    avg=np.mean(acc)
    sd=np.std(acc)
    avg_metrics.append(("Accuracy",avg,sd))
    yaml_data["Accuracy"]={"Mean":float(avg),"Standard Deviation":float(sd)}

    pla=data['PLA'].values
    avg=np.mean(pla)
    sd=np.std(pla)
    avg_metrics.append(("PLA",avg,sd))
    yaml_data["PLA"]={"Mean":float(avg),"Standard Deviation":float(sd)}

    
    prec=data['Precision'].values
    avg=np.mean(prec)
    sd=np.std(prec)
    avg_metrics.append(("Precision",avg,sd))
    yaml_data["Precision"]={"Mean":float(avg),"Standard Deviation":float(sd)}
    
    rec=data['Recall'].values
    avg=np.mean(rec)
    sd=np.std(rec)
    avg_metrics.append(("Recall",avg,sd))
    yaml_data["Recall"]={"Mean":float(avg),"Standard Deviation":float(sd)}

    f1score=data['F1-Score'].values
    avg=np.mean(f1score)
    sd=np.std(f1score)
    avg_metrics.append(("F1-Score",avg,sd))
    yaml_data["F1-Score"]={"Mean":float(avg),"Standard Deviation":float(sd)}

    path="../results/results.yaml"

    with open(path,"w") as file:
        yaml.dump(yaml_data,file)
    
    for m in avg_metrics:
        print("AVG. %s = %f SD = %f \n" %(m[0],m[1],m[2]))
        
"""
Main routine
"""
def main():
    set_seed(seed)
    metrics=[]
    for f in range(1,nfolds+1):
        print("==================================================================")
        print("FOLD",f)
        x_train,y_train=upload_data(f)
        net=upload_net(f)
        som,winmap,default_class=apply_SOM(x_train,y_train,f) 
        save_som_plot(som,x_train,y_train,f)
        save_prototype(som,x_train,f)       
        y_true,y_pred,names=predict_class(net,som,winmap,default_class,f)
        m=compute_metrics(y_true,y_pred,names,f)
        metrics.append(m)
        print("==================================================================")
    
    save_metrics(metrics)


if __name__=="__main__":
    main()