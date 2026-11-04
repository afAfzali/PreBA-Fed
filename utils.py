import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist,cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.ticker import MultipleLocator
from scipy.spatial import ConvexHull

def average_weights(w,sample_num):
    """
    w: list of lists
    sample_num :list of int 
    """
    total_sample_num=sum(sample_num)
    a=[]
    avg_w=[]
    
    for i in range(len(w)):
        t=[]
        for j in range(len(w[0])):
            t.append(w[i][j]*(sample_num[i]/total_sample_num))
        a.append(t) 
    avg_w=[sum(k) for k in zip(*a)]  
    
    return avg_w

def get_dataset(dataset,model):
    if dataset=='mnist':
        if model in ('cnn1','cnn2','cnn3'):
            X_train,Y_train,X_test,Y_test=get_mnist_cnn()
        # elif model=='mlp':
        #     X_train,Y_train,X_test,Y_test=get_mnist_mlp()
            
    elif dataset=='cifar10':
        X_train,Y_train,X_test,Y_test=get_cifar10()
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return X_train,Y_train,X_test,Y_test

def get_mnist_mlp():            
    (X_train,Y_train),(X_test,Y_test)=mnist.load_data()
    X_train=X_train.reshape(X_train.shape[0],28*28).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0],28*28).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test
    
def get_mnist_cnn():
    (X_train,Y_train),(X_test,Y_test)=mnist.load_data()
    X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test

def get_cifar10():

    (X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
    X_train=X_train.astype('float32')/255.0
    X_test =X_test.astype('float32')/255.0
    Y_train=to_categorical(Y_train.flatten(),num_classes=10)
    Y_test=to_categorical(Y_test.flatten(),num_classes=10)
    return X_train,Y_train,X_test,Y_test

def iid_split(train_data,train_label,test_data,test_label,num_clients):  
    train_size=int(len(train_data)/num_clients)
    train_idx=list(range(len(train_data)))
    test_size=int(len(test_data)/num_clients)             
    test_idx=list(range(len(test_data)))
    train_partitions=[0]*num_clients
    test_partitions=[0]*num_clients
    
    for i in range(num_clients):
        indxs=np.random.choice(train_idx,train_size,replace=False)
        train_partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[indxs],train_label[indxs]))
        train_idx=list(set(train_idx)-set(indxs)) 
    for i in range(num_clients):
        indxs=np.random.choice(test_idx,test_size,replace=False)
        test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[indxs],test_label[indxs]))
        test_idx=list(set(test_idx)-set(indxs))                    
    return train_partitions,test_partitions


def poison_data_numpy(x,y,poison_ratio,noise_mean,noise_std,label_flip):
    n=len(x)
    idx=np.random.choice(n,size=int(n*poison_ratio),replace=False)
    if label_flip:        
        num_classes=len(y[0])
        y[idx]=np.eye(num_classes)[(np.argmax(y[idx],axis=1)+np.random.randint(1,num_classes,size=len(idx)))%num_classes]
        
        # new_labels=np.eye(num_classes)[np.random.randint(0,num_classes,size=len(idx))]
        # y_p[idx]=new_labels
    else:
        # x[idx]=-x[idx]
        # x[idx]=x[idx]/255.0
        x[idx]+=np.random.normal(noise_mean,noise_std,x[idx].shape)
    return x,y

def poison_tf_dataset(dataset,poison_ratio,noise_mean,noise_std,label_flip):
    x_list,y_list=[],[]
    for image,label in dataset:
        x_list.append(image.numpy())   # (28,28,1)
        y_list.append(label.numpy())
    x_all=np.stack(x_list,axis=0)          #  (N,28,28,1)
    y_all=np.stack(y_list,axis=0)          #  (N,10)

    x_p,y_p=poison_data_numpy(x_all,y_all,poison_ratio,noise_mean,noise_std,label_flip)
    return tf.data.Dataset.from_tensor_slices((x_p,y_p))

def print_array(arr,buggy_indxs,file=None):
    str_arr=[[str(x) for x in row] for row in arr]
    cols=len(str_arr[0])  
    widths=[0]*cols
    for row in str_arr:
        for j,val in enumerate(row):
            widths[j]=max(widths[j],len(val))
    
    padding=2
    widths=[w+padding for w in widths]
    
    #for row in str_arr:
    for idx,row in enumerate(str_arr):
        if idx in buggy_indxs:
            left_cell='> '
        else:
            left_cell='  '
        line=left_cell+''.join(f'{val:<{widths[j]}}' for j,val in enumerate(row))

        if file:
            print('  '+line,file=file)
        else:
            print(line)

def extract_weights_vector(model):
    weights=model.get_weights()               
    flattened=[w.flatten() for w in weights] 
    return np.concatenate(flattened)       

def plot_tsne(itr,model_names,X,folder,perplexity,plot_name):   # بر اساس تعاد مدل هایی که میرن برای شکل باید تنظیمش کنم
    n =len(model_names)
    if n<3:
        perp=1.0
    else:
        perp=min(3,n-1)

    n_components_pca=min(50,n-1)   
    pca=PCA(n_components=n_components_pca)
    X_reduced=pca.fit_transform(X)
    
    tsne=TSNE(n_components=2,perplexity=perp,random_state=None)
    X_embedded=tsne.fit_transform(X_reduced)   

    plt.figure(figsize=(8,6))
    scatter=plt.scatter(X_embedded[:, 0],X_embedded[:, 1],c=range(len(model_names)),cmap='tab10',s=80)
    
    for i,name in enumerate(model_names):     
        plt.annotate(name,(X_embedded[i, 0],X_embedded[i, 1]),textcoords="offset points",xytext=(0,8),ha='center',fontsize=9)

    plt.title(f't-SNE iteration-{itr+1} ({plot_name})')
    plt.grid(alpha=0.3)
    #plt.gca().xaxis.set_major_locator(MultipleLocator(5))   
    #plt.gca().yaxis.set_major_locator(MultipleLocator(5))
    plt.tight_layout()
    plt.savefig(fr"./{folder}/iteration {itr+1}-{plot_name}.png",dpi=300,bbox_inches='tight')
    plt.close() 


def plot_all_rounds_tsne(all_vecs,model_names,iteration,real_buggy_clients_idxs,perplexity,folder,plot_name):
    num_models=len(model_names)    
    number_points=iteration*num_models

    X=np.stack(all_vecs,axis=0)      

    if number_points<3:
        perp=1.0
    else:
        perp=min(perplexity,number_points-1)

    # PCA
    n_pca=min(50,number_points-1)
    pca=PCA(n_components=n_pca,random_state=None)
    X_pca=pca.fit_transform(X)

    # t‑SNE
    tsne=TSNE(n_components=2,perplexity=perp,random_state=None)
    X_emb=tsne.fit_transform(X_pca)    # بعد اول : num_models × iteration     بعد دوم : مقدار محور افقی و عمودی هر کدوم 

    colors=['gray']*num_models
    colors[0]='blue' 
    
    number_buggy_clients=len(real_buggy_clients_idxs)
    highlight_indxs=[]
    distinct_colors=['red','green','orange','purple','brown','pink',
    'cyan','magenta','olive','teal','maroon','gold',
    'darkorange','lime','sienna','coral','indigo','salmon',
    'peru','plum','khaki','crimson','turquoise','tan']
    
    for i,c_idx in enumerate(real_buggy_clients_idxs):
        colors[c_idx+1]=distinct_colors[i%len(distinct_colors)]    #  تا 24 تا رنگ متفاوت می دهد واسه بیشتر ار اون :(
        highlight_indxs.append(c_idx+1)      # جای کلاینت های باگ دار در لیستی که اولش مدل تجمعی هست

    #نمودار اول: مسیر حرکت مدل‌ها (خطوط)
    plt.figure(figsize=(12,9))
    for m_idx,m_name in enumerate(model_names):
        model_point_indices=[m_idx +r*num_models for r in range(iteration)]    # اندیس های هر مدل رو پیدا می کنه 
        model_points=X_emb[model_point_indices]

        if m_idx==0 or m_idx in highlight_indxs:   # برای مدل تجمیعی و کلاینت های باگ دار (واسه راهنما نمودار )
            lbl=m_name
        else:
            lbl=None   

        plt.plot(model_points[:, 0],model_points[:, 1],marker='o',linestyle='-',color=colors[m_idx],label=lbl,linewidth=2,markersize=5)

        plt.scatter(model_points[0, 0],model_points[0, 1],color=colors[m_idx],s=100,edgecolors='black',linewidth=1.5,zorder=5)        

    plt.title(f' t-SNE over {iteration} iterations ({plot_name})')
    plt.legend(loc='best',fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fr"./{folder}/{plot_name}-all rounds tsne (line).png",dpi=300,bbox_inches='tight')
    plt.close() 
    
    plt.figure(figsize=(12, 9))
    
    point_colors = []
    for r in range(iteration):
        for m in range(num_models):
            point_colors.append(colors[m])

    plt.scatter(X_emb[:, 0],X_emb[:, 1],c=point_colors,s=40,alpha=0.85,edgecolors='none')

    for r in range(iteration):
        start=r*num_models
        end=start+num_models
        pts=X_emb[start:end]

        if num_models>=3:
            try:
                hull=ConvexHull(pts)
                hull_pts=pts[hull.vertices]
                hull_pts=np.vstack([hull_pts,hull_pts[0]])
                plt.plot(hull_pts[:, 0],hull_pts[:, 1],'k--',linewidth=1.2,alpha=0.7)
            except:
                pass
        elif num_models==2:
            plt.plot(pts[:, 0],pts[:, 1],'k--',linewidth=1.2,alpha=0.7)

        centroid=np.mean(pts,axis=0)
        plt.text(centroid[0],centroid[1],f'R{r+1}',fontsize=9,ha='center',va='center',bbox=dict(boxstyle='round,pad=0.2',facecolor='white',alpha=0.7))

    legend_handles=[]
    legend_handles.append(plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='blue',markersize=9,label='Agg model'))
    for c_idx in real_buggy_clients_idxs:
        idx_in_list=c_idx+1
        legend_handles.append(plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[idx_in_list],markersize=9,label=f'client-{c_idx+1}'))

    plt.legend(handles=legend_handles, loc='best', fontsize=10)
    plt.title(f't-SNE Rounds ({plot_name})')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fr"./{folder}/{plot_name}-all rounds tsne (scatter).png", dpi=300, bbox_inches='tight')
    plt.close() 

class BuggyClientTracker:    
    def __init__(self):
        self.metric_names=['delta_class','delta_score','ks','chi2']
        self.metrics={}
        for name in self.metric_names:
            self.metrics[name]={'TP':0,
                               'FP':0,
                               'TN':0,
                               'FN':0}
        self.final_performance_score={'TP':0,
                                      'FP':0,
                                      'TN':0,
                                      'FN':0}
    
    def update_confusion_matrix_metrics(self,buggy_clients_idxs,detected_buggy,real_buggy_indices,num_clients):
        for metric_name,detected in detected_buggy.items():  
            
            TP=0  
            FP=0  
            TN=0  
            FN=0  
            
            for c_id in range(num_clients):
                is_real_buggy=c_id in real_buggy_indices
                is_detected=c_id in detected
                
                if is_real_buggy and is_detected:
                    TP+=1
                elif not is_real_buggy and is_detected:
                    FP+=1
                elif not is_real_buggy and not is_detected:
                    TN+=1
                elif is_real_buggy and not is_detected:
                    FN+=1
               
            self.metrics[metric_name]['TP']+=TP
            self.metrics[metric_name]['FP']+=FP
            self.metrics[metric_name]['TN']+=TN
            self.metrics[metric_name]['FN']+=FN
            
        TP=FP=TN=FN=0
        for c_id in range(num_clients):
            is_real_buggy=c_id in real_buggy_indices
            is_detected=c_id in buggy_clients_idxs
    
            if is_real_buggy and is_detected:
                TP+=1
            elif not is_real_buggy and is_detected:
                FP+=1
            elif not is_real_buggy and not is_detected:
                TN+=1
            elif is_real_buggy and not is_detected:
                FN+=1
               
            self.final_performance_score['TP']+=TP
            self.final_performance_score['FP']+=FP
            self.final_performance_score['TN']+=TN
            self.final_performance_score['FN']+=FN
    
    def calculate_metrics(self,f_results):
        for metric_name in self.metric_names:
            TP=self.metrics[metric_name]['TP']
            FP=self.metrics[metric_name]['FP']
            TN=self.metrics[metric_name]['TN']
            FN=self.metrics[metric_name]['FN']
            
            accuracy=np.round((TP+TN)/(TP+FP+TN+FN),2) 
            if TP+FP>0:
                precision=np.round(TP/(TP+FP),2)
            else:
                precision="nan"  
            if TP+FN>0:   
                recall=np.round(TP/(TP+FN),2)
            else:
                recall="nan" 
            recall=TP/(TP+FN) if (TP+FN) > 0 else 0
            if precision=="nan" or recall=="nan" or (precision+recall==0):   #np.isnan(precision) or np.isnan(recall) or
                f1_score="nan"      #np.nan
            else:
                f1_score=np.round(2*(precision*recall)/(precision+recall),2)
                
            print(f" {metric_name} --->  Acc:{accuracy},  Precision:{precision},  Recall:{recall},  F1score:{f1_score}\n",file=f_results)

        TP=self.final_performance_score['TP']
        FP=self.final_performance_score['FP']
        TN=self.final_performance_score['TN']
        FN=self.final_performance_score['FN']
        accuracy=np.round((TP+TN)/(TP+FP+TN+FN),2) 
        if TP+FP>0:
            precision=np.round(TP/(TP+FP),2)
        else:
            precision="nan"  
        if TP+FN>0:   
            recall=np.round(TP/(TP+FN),2)
        else:
            recall="nan" 
        recall=TP/(TP+FN) if (TP+FN) > 0 else 0
        if precision=="nan" or recall=="nan" or (precision+recall==0):   #np.isnan(precision) or np.isnan(recall) or
            f1_score="nan"      #np.nan
        else:
            f1_score=np.round(2*(precision*recall)/(precision+recall),2)
                
        print("\n◼◼◼◼◼◼◼◼◼◼◼◼◼◼\n final evaluation (our approach) :\n",file=f_results)
        print(f" Acc:{accuracy},  Precision:{precision},  Recall:{recall},  F1score:{f1_score}\n",file=f_results)





