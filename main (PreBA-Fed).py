import numpy as np
import csv
import random
import tensorflow as tf
from client import Client
from server import Server 
from utils import get_dataset,iid_split
from tensorflow.keras.models import load_model
import time
from datetime import datetime
import jdatetime
import os


# ---------------- Fl and model setting
dataset="cifar10"          # cifar10
image_shape=(32,32,3)    # (32,32,3) for cifar10  //(28,28,1)  mnist
num_classes=10 
model="cnn2"    #for cnn2, cnn3
batch_size=32
iteration=30      
epochs=10                  
num_clients=10
lr=0.001
loss="categorical_crossentropy"     
metrics=["accuracy"]
verbose=0
# --------------------------------------------------
# ---------------- Differniatial testing setting
alpha=0.05/num_clients   # 0.05 
theta=0.01      #0.001   0.01
#K=3
outlier_threshold=2
# --------------------------------------------------
# ---------------- bug setting
poison_ratio=0.3      # for label flipping
#noise_std=0.1    # for data noise
label_flip=False
lr_type='too_high'      # 'too_high', 'too_low', 'negative'
#std=0.1   # for layer noise
# --------------------------------------------------

optimizer=tf.keras.optimizers.SGD(learning_rate=lr)

bug_code_list={
    0:'data bug',
    1:'learning rate bug',
    2:'layer weight noise',
    3:'activation function bug (sigmoid)'}

model_config={
    'dataset':dataset,
    'model':model,
    'loss':loss,
    'metrics':metrics,
    'lr':lr,
    'batch_size':batch_size,
    'image_shape':image_shape,
    'num_classes':num_classes}

# =============================================================================
#                         🟩          partitioning        🟩
# =============================================================================

X_train,Y_train,X_test,Y_test=get_dataset(dataset,model) 
X_server_test,Y_server_test=X_test[:10000],Y_test[:10000]
# print("X_test shape:",X_server_test.shape)        # باید (10000,28,28,1) باشه
# print("X_test[0] shape:",X_server_test[0].shape)    # باید (28,28,1) باشه
# print("X_test[0, :, :, 0] shape:",X_server_test[0, :, :, 0].shape)

print('1 : iid setting \n'
      '2 : .....? \n')
flag=int(input('select a number:')) 
print('\n')


if flag==1:       #   ...                        
    clients=[]
    train_partitions,_=iid_split(X_train,Y_train,X_test,Y_test,num_clients)
    for i in range(num_clients):
        clients.append(Client(train_partitions[i],model_config)) 
    server=Server(X_server_test,Y_server_test,model_config)   
    
del X_train,Y_train,X_test,train_partitions

#====================================================================================
now=datetime.now()
shamsi_date=jdatetime.datetime.fromgregorian(datetime=now)
folder=f"FeDebug Results\FeDebug {shamsi_date.strftime(f'%Y-%m-%d . . . %H-%M')}"
os.makedirs(folder)

server.save_model(folder)   

with open(fr"./{folder}/results.txt","w") as f_results:  
    print(f" dataset: {dataset}\n num_classes: {num_classes}\n lr: {lr}",file=f_results)
    print(f" model: {model}\n batch_size: {batch_size}\n iteration: {iteration}\n epochs: {epochs}",file=f_results)
    print(f" bug(0): {bug_code_list[0]}\n bug(1): {bug_code_list[1]}\n bug(3): {bug_code_list[3]}",file=f_results)  #\n bug(2): {bug_code_list[2]}   \n K: {K}
    print(f" alpha: {alpha}\n theta: {theta}\n outlier_threshold: {outlier_threshold}",file=f_results)
    print(f" poison_ratio: {poison_ratio}\n label_flip: {label_flip}\n lr_type: {lr_type}",file=f_results)  #\n std: {std}    \n noise_std: {noise_std}
    print(f" num_clients: {num_clients}\n flag: {flag}",file=f_results)
    print("============================================================================",file=f_results)

    for c_idx,client in enumerate(clients):
        l=[0]*num_classes
        for i,j in client.raw_dataset:
            l[np.argmax(j.numpy())]+=1
        print(f' client-{c_idx+1}:',file=f_results)
        print(f' {l}-->',end=' ',file=f_results)
        print(' ',np.sum(l),file=f_results)
        res=[idx for idx,val in enumerate(l) if val!=0]
    
        print(' ',res,file=f_results)
        print("-----------------------------------------------------------",file=f_results)
# =============================================================================
#                              🟩   apply bug     🟩
# =============================================================================

buggy_clients={}      # key:client numebr, value:bug_code(0-3)
file_path="./data.csv"

with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    print("============================================================================",file=f_results)
    print(" \n🟩 real buggy clients:",end=' ',file=f_results)
    print(" 🟩 real buggy clients:",end=' ')
    with open(file_path,'r') as file:
        reader=csv.reader(file)
        next(reader)
        next(reader)
        for row in reader:
            print(f'client-{row[0]}',end='   ',file=f_results)
            print(f'client-{row[0]}',end='   ')
            c_idx=int(row[0])-1          # first column:client number
            b_code=int(row[1])            # second column:bug_code(0-3)
            buggy_clients[c_idx]=b_code
    print('\n',file=f_results)
    print('\n')

# =============================================================================
#                          🟩          without debugging      🟩
# =============================================================================
with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼",file=f_results)
    print("                            without debugging                            ",file=f_results)
print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼")
print("                        without debugging                            ")
server.load_model(folder)
server.test_acc.clear()
for c_idx in buggy_clients.keys():
    clients[c_idx].test_acc.clear()
for itr in range(iteration): 
    print('===========================================================')
    print(f'                      iteration {itr+1} ')
    print('===========================================================')
    
    server.refresh_server()   
    
    for c_idx,client in enumerate(clients):
        server.send_to_client(client) 

        print(f'client-{c_idx+1} is training locally')
        client.local_model_train(epochs,verbose)

    # for c_idx in buggy_clients.keys():
    #     clients[c_idx].test(server.test_data)
            
        # if c_idx in buggy_clients and buggy_clients[c_idx]==2:
        #     noise=layer_noises[c_idx]
        #     if noise is not None:
        #         client.inject_bug(2,noise=noise)
        
    for client in clients:
        server.client_registering(client)
        
    # with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    #     print(f'\n🟪 aggregation on:',file=f_results)
    #     print(f'\n🟪 aggregation on:')
    #     for c_idx in list(range(num_clients)): 
    #         print(f'C-{c_idx+1}, ',end='',file=f_results)
    #         print(f'C-{c_idx+1}, ',end='')
        
    server.aggregate()
    server.test()
    
with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼\nglobal test accuracy (FedAvg) --->" ,*server.test_acc,sep="   ",file=f_results)
# print("\nglobal test accuracy (without debugging) --->" ,*server.test_acc,sep="   ")
    for c_idx in buggy_clients.keys():
        print(f"\nlocal test accuracy (client-{c_idx+1}) --->" ,*clients[c_idx].test_acc,sep="   ",file=f_results)
    

