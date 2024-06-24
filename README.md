# Classification of histopathological images using a granular computing approach. 

To reproduce our experiments, you can follow these steps:

- download the BreakHis dataset at https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/ and locate it inside the ablation study and experiment folders.



## Ablation Study

Run the script classify_som_resnet.py to:

- Obtain the embeddings of the training fragments using a ResNet152. 
- Cluster the training fragments using a Self-Organizing Map (SOM)  and then classify the testing fragments using the trained SOM. 



## Exsperiment



1. Run the script create_emb.py to train a triplet network and create the embeddings of the training fragments. 
2. Run the script classify_som.py to cluster the training fragments using a Self-Organizing Map (SOM), classify the testing fragments using the trained SOM  and produce the explanation for the testing images. 

To use this model cite:
'''
@article{amato2024explainable,
  title={Explainable Histopathology Image Classification with Self-organizing Maps: A Granular Computing Perspective},
  author={Amato, Domenico and Calderaro, Salvatore and Lo Bosco, Giosu{\'e} and Rizzo, Riccardo and Vella, Filippo},
  journal={Cognitive Computation},
  pages={1--21},
  year={2024},
  publisher={Springer}
}
'''
