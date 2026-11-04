from model.cnn import CNN_1,CNN_2,CNN_3

def create(dataset,model,loss,metrics,lr,image_shape,num_classes):  
    if model=='cnn1':
        return CNN_1(loss,metrics,lr,image_shape,num_classes)
    elif model=='cnn2':
        return CNN_2(loss,metrics,lr,image_shape,num_classes)
    elif model=='cnn3':
        return CNN_3(loss,metrics,lr,image_shape,num_classes)
        
