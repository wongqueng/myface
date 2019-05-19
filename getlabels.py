import os

andy_path="D:/PycharmProjects/myface/train/AndyLou/"
daniel_path="D:/PycharmProjects/myface/train/DanielWu/"
with open("train/data.txt","w") as f:
    ls = os.listdir(andy_path)
    for i in ls:
        f.write(andy_path+str(i)+" 0"+"\n")
    ls1 = os.listdir(daniel_path)
    for j in ls1:
        f.write(daniel_path+str(j)+" 1"+"\n")



