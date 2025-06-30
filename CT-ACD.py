import torch
import json
import numpy as np
import os
import glob

import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet152_Weights
from PIL import Image

device = torch.device("cuda")

##Set Up Time 
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

##Load Dataset TopImages(ImageNetV2)
datasets.ImageFolder(root='imagenetv2-top-images-format-val')

##Obtain Labels
with open('imagenet_class_index.json') as json_file:
    labels = json.load(json_file)

##Load ResNet Models
resnet_18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
resnet_34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).cuda()
resnet_50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
resnet_152 = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).cuda()

##Confidence Thresholds of Each model (Buruah et al.)
confThreshold_18 = 0.890
confThreshold_34 = 0.895
confThreshold_50 = 0.896
confThreshold_152 = 0.910

##Transform Function to Convert to Tensors
transform = transforms.Compose([           
 transforms.Resize(256),                  
 transforms.CenterCrop(224),              
 transforms.ToTensor(),   
 transforms.Normalize(                   
 mean=[0.485, 0.456, 0.406],             
 std=[0.229, 0.224, 0.225]                    
)
 ])

##Put Model into Eval Mode
resnet_18.eval()
resnet_34.eval()
resnet_50.eval()
resnet_152.eval()

##Open Results Text Files
resultsA = open("results-a.txt","a")
resultsB = open("results-b.txt","a")
resultsC = open("results-c.txt","a")
resultsD = open("results-d.txt","a")
resultsDeterministic = open("results-deterministic.txt","a")
stats=open("stats.txt","a")
idkRate = open("bIdkRate.txt","a")

##IDK Classifier A (ResNet-18)
def classifierA(img, threshold):
    ##Turn Image into Tensor
    val = transform(img)
    ##Change Dimensions of Tensor
    val = torch.unsqueeze(val,0)
    val = val.to(torch.device('cuda'))
    with torch.no_grad():
        ##Identify Highest Probability and Record Time
        start.record()
        prob = F.softmax(resnet_18(val),dim=1)
        end.record()
        ##Wait for Everything to Finish Running
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
    confidenceLevel, top_class = prob.topk(1,dim=1)
    return (labels[str(int(top_class))][1],float(confidenceLevel),time)

##IDK Classifier B (ResNet-34)
def classifierB(img, threshold):
    val = transform(img)
    val = torch.unsqueeze(val,0)
    val = val.to(torch.device('cuda'))
    with torch.no_grad():
        start.record()
        prob = F.softmax(resnet_34(val),dim=1)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
    confidenceLevel, top_class = prob.topk(1,dim=1)
    return (labels[str(int(top_class))][1],float(confidenceLevel),time)

##IDK Classifier C (ResNet-50)
def classifierC(img, threshold):
    val = transform(img)
    val = torch.unsqueeze(val,0)
    val = val.to(torch.device('cuda'))
    with torch.no_grad():
        start.record()
        prob = F.softmax(resnet_50(val),dim=1)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
    confidenceLevel, top_class = prob.topk(1,dim=1)
    return (labels[str(int(top_class))][1],float(confidenceLevel),time)

##IDK Classifier D (ResNet-152)
def classifierD(img, threshold):
    val = transform(img)
    val = torch.unsqueeze(val,0)
    val = val.to(torch.device('cuda'))
    with torch.no_grad():
        start.record()
        prob = F.softmax(resnet_152(val),dim=1)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
    confidenceLevel, top_class = prob.topk(1,dim=1)
    return (labels[str(int(top_class))][1],float(confidenceLevel),time)

statsTime=0
countA=0
Acorrect=0
countB=0
Bcorrect=0
countC=0
Ccorrect=0
countD=0
Dcorrect=0
evaluationNum=0

##IDK Cascade Optimal Cascade for a specified Classification Threshold (A,B,D) Threshold=0.65
def idkCascade(img, correctClass):
    
    val = "Incorrect"
    totalTime = 0
    validation = labels[correctClass][1]
    global statsTime
    print(statsTime)
    global evaluationNum
    AClass, AConfidence, ATime = classifierA(img,confThreshold_18)
    totalTime+=ATime
    if ((AConfidence<confThreshold_18)==False):
        if(AClass==validation):
            evaluationNum+=1
            global Acorrect
            Acorrect+=1
            val="Correct"
        statsTime+=totalTime
        resultsA.write(val+': '+AClass+' '+str(AConfidence)[:5]+' '+str(ATime)[:5]+'ms\n')
        global countA
        countA+=1
        
        return 
    
    CClass, CConfidence, CTime = classifierC(img,confThreshold_50)
    totalTime+=CTime
    if ((CConfidence<confThreshold_50)==False):
        if(CClass==validation):
            evaluationNum+=1
            global Ccorrect
            Ccorrect+=1
            val="Correct"
        statsTime+=totalTime
        resultsC.write(val+'\n'+AClass+' '+str(AConfidence)[:5]+' '+str(ATime)[:5]+'ms\n'+CClass+' '+str(CConfidence)[:5]+' '+str(CTime)[:5]+'ms\n'+str(totalTime)[:5]+'ms\n\n')
        global countC
        countC+=1
        idkRate.write(str(AConfidence)[:5]+" "+CClass+"\n")
        return 
    
    idkRate.write(str(AConfidence)[:5]+" IDK\n")
    DClass, DConfidence, DTime = classifierD(img,confThreshold_152)
    totalTime+=DTime
    if(DClass==validation):
        evaluationNum+=1
        global Dcorrect
        Dcorrect+=1
        val="Correct"
    statsTime+=totalTime
    resultsD.write(val+'\n'+AClass+' '+str(AConfidence)[:5]+' '+str(ATime)[:5]+'ms\n'+CClass+' '+str(CConfidence)[:5]+' '+str(CTime)[:5]+'ms\n'+DClass+' '+str(DConfidence)[:5]+' '+str(DTime)[:5]+'ms\n'+str(totalTime)[:5]+'ms\n\n')
    global countD
    countD+=1

    return 
    


##Use PIL to open Images
img = Image.open('imagenetv2-top-images-format-val/0/0af3f1b55de791c4144e2fb6d7dfe96dfc22d3fc.jpeg')

Class, Confidence, Time = classifierA(img,confThreshold_18)
Class, Confidence, Time = classifierB(img,confThreshold_34)
Class, Confidence, Time = classifierC(img,confThreshold_50)
Class, Confidence, Time = classifierD(img,confThreshold_152)


##Run Through Folder
directory = r"imagenetv2-top-images-format-val"

index = 0
for path, folders, files in os.walk(directory):
    # Open file
    for folder_name in folders:
        path = os.path.join(directory,folder_name)
        print('Folder: '+str(folder_name))
        for name in os.listdir(os.path.join(directory,folder_name)):
            img = Image.open(os.path.join(path,name).replace("\\","/"))
            idkCascade(img, str(folder_name))
    break

stats.write("Average Time: "+str(statsTime/10000)[:5]+"ms"+
            "\nClassification Accuracy: "+str((evaluationNum/10000)*100)[:5]+
            "\n\nClassifier A Count: "+str(countA)+"\nAccuracy: "+str((Acorrect/countA)*100)[:5]+"%"+
            "\n\nClassifier C Count: "+str(countC)+"\nAccuracy: "+str((Ccorrect/countC)*100)[:5]+"%"+
            "\n\nClassifier D Count: "+str(countD)+"\nAccuracy: "+str((Dcorrect/countD)*100)[:5]+"%")

resultsA.close()
resultsB.close()
resultsC.close()
resultsD.close()
stats.close()
resultsDeterministic.close()
idkRate.close()