#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf 
import numpy as np
import pandas as pd
from utils import average_weights,print_array
from model.initialize_model import create
from scipy.stats import ks_2samp,chi2_contingency
import statistics as sta
from tensorflow.keras.models import clone_model




class Server:          
    def __init__(self,X_test,Y_test,model_config):         
        self.buffer=[]
        self.participated_sample=[]
        self.test_data=tf.data.Dataset.from_tensor_slices((X_test,Y_test))
        self.test_acc=[]
        self.model=create(
            model_config['dataset'],
            model_config['model'],
            model_config['loss'],
            model_config['metrics'],
            model_config['lr'],
            model_config['image_shape'],
            model_config['num_classes'])
        self.tempt_model=clone_model(self.model)

    def save_model(self,folder):
        self.model.save_weights(fr".\{folder}\initial-global-model.h5")   
    
    def load_model(self,folder):
        self.model.load_weights(fr".\{folder}\initial-global-model.h5") 
        
    def send_to_client(self,client): 
        client.model.set_weights(self.model.get_weights()) 

    def test(self):
        _,acc=self.model.evaluate(self.test_data.batch(32),verbose=0)
        self.test_acc.append(np.round(acc,2))
        
    def refresh_server(self):          
        self.buffer.clear()                 
        self.participated_sample.clear() 

    def predict_with_temp_model(self,client):
        self.tempt_model.set_weights(client.model.get_weights())
        prob_predict=self.tempt_model.predict(self.test_data.batch(32),verbose=0)
        label_predict=np.array([np.argmax(y) for y in prob_predict])        
        return prob_predict,label_predict

    def differential_debug(self,clients,num_classes,alpha,theta,outlier_threshold,itr,buggy_indxs,f_results):  #K
        """ identify buggy clients 
          returns: list of buggy clients idxs """
        
        labels={}   # label predictions       (num_sample_of_sever_test,)
        probs={}    # prob predictions  (num_samples_of_sever_test,num_classes)
        metric_names=['delta_class','delta_score','ks','chi2']
        detected_buggy_per_metric={
                        'delta_class':[],
                        'delta_score':[],
                        'ks':[],
                        'chi2':[]}  #  
        
        for c_idx,client in enumerate(clients):
            probs[c_idx],labels[c_idx]=self.predict_with_temp_model(client)
            
        num_clients=len(clients)
        A_class=np.zeros((num_clients,num_clients),dtype=int)
        A_score=np.zeros((num_clients,num_clients),dtype=int)
        A_ks= np.zeros((num_clients,num_clients),dtype=int)
        A_chi2=np.zeros((num_clients,num_clients),dtype=int)
            
        for c_i in range(num_clients):
            for c_j in range(c_i+1,num_clients):
                
                # delta class: count of different predicted labels
                class_diff=(labels[c_i]!=labels[c_j]).sum()
                A_class[c_i,c_j]=int(class_diff)
                A_class[c_j,c_i]=int(class_diff)

                ### delta score: count of (sample,class) where prob difference>theta
                score_diff=(np.abs(probs[c_i]-probs[c_j])>theta).sum()         
                A_score[c_i,c_j]=score_diff
                A_score[c_j,c_i]=score_diff

                ### KS test for each class
                ks_count=0                      
                for l in range(num_classes):
                    probs_ci_l=probs[c_i][:,l]   #نمونه ها کمتر بشه    
                    probs_cj_l=probs[c_j][:,l]
                    _,p_value=ks_2samp(probs_ci_l,probs_cj_l,method='asymp')
                    if p_value<alpha:
                        ks_count+=1
                A_ks[c_i,c_j]=ks_count
                A_ks[c_j,c_i]=ks_count
    
                ### Chi2 test on label predictions  
                table=pd.crosstab(labels[c_i],labels[c_j])
                chi2,_,_,_=chi2_contingency(table)     
                A_chi2[c_i,c_j]=chi2
                A_chi2[c_j,c_i]=chi2
                    
        print("\n============================================================================",file=f_results)
        print(f'                             iteration {itr+1} ',file=f_results)
        print("============================================================================",file=f_results)
        print("delta class matrix:\n\n",file=f_results)
        print_array(A_class,buggy_indxs,file=f_results)
        print("\n\ndelta score matrix:\n\n",file=f_results)
        print_array(A_score,buggy_indxs,file=f_results)
        print("\n\nks matrix:\n\n",file=f_results)
        print_array(A_ks,buggy_indxs,file=f_results)
        print("\n\nchi2 matrix:\n\n",file=f_results)
        print_array(A_chi2,buggy_indxs,file=f_results)

        """  
        ### A_diff: average of each metric over c_i!=c_j
        A_diff=np.zeros((num_clients,4))
        for c_i in range(num_clients):
            sum_class=sum_score=sum_ks=sum_chi2=0.0
            for c_j in range(num_clients):
                if c_i==c_j:
                    continue
                sum_class+=A_class[c_i,c_j]
                sum_score+=A_score[c_i,c_j]
                sum_ks+=A_ks[c_i,c_j]
                sum_chi2+=A_chi2[c_i,c_j]
            A_diff[c_i,0]=sum_class/(num_clients-1)
            A_diff[c_i,1]=sum_score/(num_clients-1)
            A_diff[c_i,2]=sum_ks/(num_clients-1)
            A_diff[c_i,3]=sum_chi2/(num_clients-1)
            
        ### compute column-wise mean and std
        mu=np.zeros(4)
        sigma=np.zeros(4)
        for col in range(4):
            col_vals=A_diff[:,col] 
            mu[col]=col_vals.mean()          # num_clients?
            sigma[col]=np.std(col_vals,ddof=0)      # num_clients?

        """
        ### A_median: median of each metric over c_i!=c_j
        A_median=np.zeros((num_clients,4))
        for c_i in range(num_clients):
            delta_class_row_without_diag=[A_class[c_i][c_j] for c_j in range(num_clients) if c_j!=c_i]
            delta_score_row_without_diag=[A_score[c_i][c_j] for c_j in range(num_clients) if c_j!=c_i]
            ks_row_without_diag=[A_ks[c_i][c_j] for c_j in range(num_clients) if c_j!=c_i]
            chi2_row_without_diag=[A_chi2[c_i][c_j] for c_j in range(num_clients) if c_j!=c_i]

            A_median[c_i,0]=sta.median(delta_class_row_without_diag)   
            A_median[c_i,1]=sta.median(delta_score_row_without_diag)
            A_median[c_i,2]=sta.median(ks_row_without_diag)
            A_median[c_i,3]=sta.median(chi2_row_without_diag)
            
        ### compute column-wise lower_bound , upper_bound and iqr
        lower_bound=np.zeros(4)
        upper_bound=np.zeros(4)

        for col in range(4):
            col_vals=A_median[:,col] 
            q1=np.percentile(col_vals,25)
            q3=np.percentile(col_vals,75)
            iqr=q3-q1
            lower_bound[col]=q1-(iqr*1.5)
            upper_bound[col]=q3+(iqr*1.5)

        print('-----------------------------------------------------------',file=f_results)
        print("\nidentifying buggy clients using .... :\n",file=f_results)
        d_class,d_score,ks,chi2=[],[],[],[]
        ### find outliers 
        A_outcount=np.zeros(num_clients,dtype=int)
        for col in range(4):
            for c_i in range(num_clients):
                if (A_median[c_i,col]<lower_bound[col]) | (A_median[c_i,col]>upper_bound[col]):     
                    A_outcount[c_i]+=1
                    detected_buggy_per_metric[metric_names[col]].append(c_i)
                    if col==0:
                        d_class.append(f'client-{c_i+1}')
                    elif col==1:
                        d_score.append(f'client-{c_i+1}')
                    elif col==2:
                        ks.append(f'client-{c_i+1}')
                    else:
                        chi2.append(f'client-{c_i+1}')
        print("delta class --->" ,*d_class,sep="  ",file=f_results)
        print("delta score --->" ,*d_score,sep="  ",file=f_results)
        print("KS test --->" ,*ks,sep="  ",file=f_results)
        print("Chi2 test --->" ,*chi2,sep="  ",file=f_results)
        buggy_clients_idxs=[]
        for c_idx in range(num_clients):
            if A_outcount[c_idx]>=outlier_threshold:
                buggy_clients_idxs.append(c_idx)
                clients[c_idx].counter+=1

        for c_idx in buggy_clients_idxs:  
            if clients[c_idx].counter<(itr+1)/2:
                buggy_clients_idxs=list(set(buggy_clients_idxs)-{c_idx})
            
        return buggy_clients_idxs,detected_buggy_per_metric

    def client_registering(self,client):
        self.buffer.append(client.model.get_weights()) 
        self.participated_sample.append(client.train_num) 

    def aggregate(self): 
        self.model.set_weights(average_weights(self.buffer,self.participated_sample))

