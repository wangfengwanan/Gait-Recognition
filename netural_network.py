import numpy as np
with open('x_train.txt') as fp1:
    a1 = fp1.read().split()
    print(len(a1))
    x_train_datas = []
    for i in range(7767):
        x_train_data = a1[561*i:561*(i+1)]
        x_train_data = list(map(lambda x:np.float32(x),x_train_data))
        x_train_datas.append(x_train_data)
    print(len(x_train_datas))
    
with open('y_train.txt') as fp3:
    y_train_datas = fp3.read().split()
    y_train_datas = list(map(lambda x:np.float32(x),y_train_datas))
    j=0
    for i in y_train_datas[:]:
        a = [0,0,0,0,0,0,0,0,0,0,0,0]
        a[int(i)-1] = 1
        y_train_datas[j] = a
        j += 1
    print(y_train_datas[:10])
