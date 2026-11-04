import numpy as np
import csv
import random
import tensorflow as tf
from client import Client
from server import Server 
from utils import BuggyClientTracker
from utils import get_dataset,iid_split,extract_weights_vector,plot_tsne,plot_all_rounds_tsne
from tensorflow.keras.models import load_model
import time
from datetime import datetime
import jdatetime
import os


# ---------------- Fl and model setting
dataset="cifar10"          # cifar10  mnist
image_shape=(32,32,3)   # (32,32,3) for cifar10  (28,28,1)
num_classes=10
model="cnn3"    #for cnn2, cnn3  "cnn1"
batch_size=32
iteration=10    
epochs=20                   
num_clients=10
lr=0.0005  #  0.0005
loss="categorical_crossentropy"     
metrics=["accuracy"]
verbose=0
# --------------------------------------------------
# ---------------- Differniatial testing setting
alpha=0.05/num_clients   # 0.05 
theta=0.01      #0.001   0.01
#K=3
outlier_threshold=2
perplexity=40  # for t-sne
# --------------------------------------------------
# ---------------- bug setting
poison_ratio=1   # for label flipping or data bug
noise_mean=0.2  # for data noise
noise_std=0.5     # for data noise
label_flip=True
lr_type='negative'      # options: 'too_high', 'too_low', 'negative'
# --------------------------------------------------

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


if flag==1:                         
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
    print(f" noise_mean: {noise_mean}\n noise_std: {noise_std}",file=f_results)
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
#                          🟩          FedAvg without buggy clients      🟩
# =============================================================================
with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼",file=f_results)
    print("                            FedAvg without buggy clients                            ",file=f_results)
print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼")
print("                        FedAvg without buggy clients                            ")
server.load_model(folder)
server.test_acc.clear()

clients_idxs=list(range(num_clients))

for itr in range(iteration): 
    print('===========================================================')
    print(f'                      iteration {itr+1} ')
    print('===========================================================')
    
    server.refresh_server()   
    
    for c_idx in clients_idxs:
        server.send_to_client(clients[c_idx]) 

        print(f'client-{c_idx+1} is training locally')
        clients[c_idx].local_model_train(epochs,verbose)
            
    for c_idx in clients_idxs:
        server.client_registering(clients[c_idx])
        
    server.aggregate()
    server.test()
    
with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    print("\nglobal test accuracy(FedAvg without bug)  --->" ,*server.test_acc,sep="   ",file=f_results)
    
# =============================================================================
#                              🟩   apply bug     🟩
# =============================================================================

real_buggy_clients={}      # key:client numebr, value:bug_code(0-3)
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
            real_buggy_clients[c_idx]=b_code
    print('\n',file=f_results)
    print('\n')
    
    for c_idx,bug_code in real_buggy_clients.items():
        if bug_code==0:   # data bug
            print(f'client-{c_idx+1}',end='',file=f_results)
            print(f'client-{c_idx+1}',end='')
            clients[c_idx].inject_bug(bug_code,poison_ratio=poison_ratio,noise_mean=noise_mean,noise_std=noise_std,
                                                                               label_flip=label_flip,f_results=f_results)
        elif bug_code==1:        # learning rate bug
            print(f'client-{c_idx+1}',end='',file=f_results)
            print(f'client-{c_idx+1}',end='')
            clients[c_idx].inject_bug(bug_code,lr_type=lr_type,f_results=f_results)
        
        elif bug_code==3:        # activation function bug(sigmoid)
            print(f'client-{c_idx+1}',end='',file=f_results)
            print(f'client-{c_idx+1}',end='')
            clients[c_idx].inject_bug(bug_code,f_results=f_results)
            
# =============================================================================
#                          🟩          FeDebug        🟩
# =============================================================================

with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼",file=f_results)
    print("                              PreBA-Fed                            ",file=f_results)
print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼")
print("                               PreBA-Fed                            ")
server.load_model(folder)
server.test_acc.clear()
all_weight_vectors=[]  # برای t-sne
model_names=["Agg model"]   # برای t-sne
# for c_idx in real_buggy_clients.keys():
#     model_names.append(f'client-{c_idx+1}') 
for c_idx in range(num_clients):
    model_names.append(f'client-{c_idx+1}') 

tracker=BuggyClientTracker()
for itr in range(iteration): 
    print('===========================================================')
    print(f'                      iteration {itr+1} ')
    print('===========================================================')
    server.refresh_server()   
    
    for c_idx,client in enumerate(clients):
        server.send_to_client(client) 

        print(f'client-{c_idx+1} is training locally')
        client.local_model_train(epochs,verbose)

    for c_idx in real_buggy_clients.keys():
        clients[c_idx].test(server.test_data)

    with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
        buggy_clients_idxs,detected_buggy_per_metric=server.differential_debug(clients,num_classes,alpha,theta,outlier_threshold,itr,list(real_buggy_clients.keys()),f_results)
        
        tracker.update_confusion_matrix_metrics(buggy_clients_idxs,detected_buggy_per_metric,list(real_buggy_clients.keys()),num_clients)
 
        # print(f'\n🟪 buggy clients detected:(outlier_threshold: {outlier_threshold})')
        print(f'\n🟪 buggy clients detected:(outlier_threshold: {outlier_threshold})',file=f_results)
        for c_idx in buggy_clients_idxs:
            # print(f'  client-{c_idx+1}',end='')       
            print(f'  client-{c_idx+1}',end='',file=f_results)
            
        
    all_clients_idxs=list(range(num_clients))
    safe_clients_idxs=list(set(all_clients_idxs)-set(buggy_clients_idxs))
    for c_idx in safe_clients_idxs:
        server.client_registering(clients[c_idx])
            
    server.aggregate()
    server.test()

    ####   t-sne
    vec=extract_weights_vector(server.model)
    all_weight_vectors.append(vec) 
    for c_idx in range(num_clients):
        vec=extract_weights_vector(clients[c_idx].model)
        all_weight_vectors.append(vec) 
plot_all_rounds_tsne(all_weight_vectors,model_names,iteration,list(real_buggy_clients.keys()),perplexity,folder,plot_name="PreBA-Fed")
    ####
with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼\nglobal test accuracy (PreBA-Fed) --->",*server.test_acc,sep="   ",file=f_results)
    for c_idx in real_buggy_clients.keys():
        print(f"\nlocal test accuracy (client-{c_idx+1}) --->" ,*clients[c_idx].test_acc,sep="   ",file=f_results)
    print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼\n metrics :\n",file=f_results)
    tracker.calculate_metrics(f_results)    

# =============================================================================
#                          🟩          without debugging      🟩
# =============================================================================
with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼",file=f_results)
    print("                            FedAvg with buggy clients                            ",file=f_results)
print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼")
print("                        FedAvg with buggy clients                            ")
server.load_model(folder)
server.test_acc.clear()
for c_idx in real_buggy_clients.keys():
    clients[c_idx].test_acc.clear()
    
all_weight_vectors.clear()
for itr in range(iteration): 
    print('===========================================================')
    print(f'                      iteration {itr+1} ')
    print('===========================================================')
    
    server.refresh_server()   
    
    for c_idx,client in enumerate(clients):
        server.send_to_client(client) 

        print(f'client-{c_idx+1} is training locally')
        client.local_model_train(epochs,verbose)

    for c_idx in real_buggy_clients.keys():
        clients[c_idx].test(server.test_data)
        
    for client in clients:
        server.client_registering(client)
        
    server.aggregate()
    server.test()

    ####   t-sne
    vec=extract_weights_vector(server.model)
    all_weight_vectors.append(vec) 
    for c_idx in range(num_clients):
        vec=extract_weights_vector(clients[c_idx].model)
        all_weight_vectors.append(vec) 
plot_all_rounds_tsne(all_weight_vectors,model_names,iteration,list(real_buggy_clients.keys()),perplexity,folder,plot_name="FedAvg-with bug")
    
with open(fr"./{folder}/results.txt","a",encoding="utf-8") as f_results:  
    print("\nglobal test accuracy (FedAvg with bug) --->" ,*server.test_acc,sep="   ",file=f_results)
# print("\nglobal test accuracy (without debugging) --->" ,*server.test_acc,sep="   ")
    for c_idx in real_buggy_clients.keys():
        print(f"\nlocal test accuracy (client-{c_idx+1}) --->" ,*clients[c_idx].test_acc,sep="   ",file=f_results)
