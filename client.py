import numpy as np
import tensorflow as tf 
from model.initialize_model import create
from utils import poison_tf_dataset

class Client:    
    def __init__(self,train_partition,model_config):
        """
        train_partition: tf.data.Dataset (unbatched)
        """
        self.raw_dataset=train_partition
        self.train_num=train_partition.cardinality().numpy()
        self.batch_size=model_config['batch_size']
        self.base_lr=model_config['lr'] 
        self.test_acc=[]
        self.counter=0   
        
        train_partition=train_partition.shuffle(self.train_num)
        self.train_data=train_partition.batch(self.batch_size,drop_remainder=False)
        self.model=create(
            model_config['dataset'],
            model_config['model'],
            model_config['loss'],
            model_config['metrics'],
            model_config['lr'],
            model_config['image_shape'],
            model_config['num_classes'])    
        
    def local_model_train(self,epochs,verbose):  
        self.model.fit(self.train_data,epochs=epochs,verbose=verbose) 

    # def prob_and_label_predict(self,server_test_data):
    #     x=[]
    #     for i,_ in server_test_data:
    #         x.append(i.numpy())
    #     prob_predict=self.model.predict(np.array(x),verbose=0)
    #     label_predict=np.array([np.argmax(y) for y in prob_predict])
    #     return prob_predict,label_predict

    def test(self,test_server_data):
        _,acc=self.model.evaluate(test_server_data.batch(32),verbose=0)
        self.test_acc.append(np.round(acc,2))

    def inject_bug(self,bug_code,**kwargs):
        """
        bug_code:
        0:data bug
        1:learning rate bug
        2:layer weight noise(first Conv2 layer)    فعلا حذف
        3:activation function bug(sigmoid)
        """
        if bug_code==0:
            self.apply_data_bug(**kwargs)
        elif bug_code==1:
            self.apply_lr_bug(**kwargs)
        # elif bug_code==2:
        #     self.apply_layer_bug(**kwargs)
        elif bug_code==3:
            self.apply_activation_bug(**kwargs) 
        else:
            print(f"🧨 Warning:unknown bug code {bug_code},no injection.")

    def apply_data_bug(self,poison_ratio,noise_mean,noise_std,label_flip,f_results):  
        poisoned_data=poison_tf_dataset(self.raw_dataset,poison_ratio,noise_mean,noise_std,label_flip)
        poisoned_data=poisoned_data.shuffle(self.train_num)
        self.train_data=poisoned_data.batch(self.batch_size,drop_remainder=False)
        print(f" ---> Data poisoned (ratio={poison_ratio}, label_flip={label_flip})")
        print(f" ---> Data poisoned (ratio={poison_ratio}, label_flip={label_flip})",file=f_results)
        
    def apply_lr_bug(self,lr_type,f_results):
        base_lr=self.base_lr
        if lr_type=='too_high':
            new_lr=base_lr*1000.0
        elif lr_type=='too_low':
            new_lr= base_lr*0.001 #1e-3   
        elif lr_type=='negative':
            new_lr=-base_lr  
        self.model.optimizer.learning_rate.assign(new_lr)
        print(f" ---> Learning rate changed to {new_lr} (was {base_lr})")
        print(f" ---> Learning rate changed to {new_lr} (was {base_lr})",file=f_results)
     
    def apply_activation_bug(self,f_results):
        last_layer=self.model.layers[-1]
        last_layer.activation=tf.keras.activations.sigmoid
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.base_lr),loss="categorical_crossentropy",metrics=["accuracy"])
        print(f" ---> Last layer activation changed to sigmoid")
        print(f" ---> Last layer activation changed to sigmoid",file=f_results)





