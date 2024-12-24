import matplotlib.pyplot as plt
import a_new_test,os,matplotlib
import torchvision.models as models
import torchvision, torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.models import ResNet152_Weights, resnet152

def main():
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  #경로 설정
  img_dir=r'E:\Datasets\food-101\images'
  train_list=r"E:\Datasets\food-101\meta\train.json"
  test_list=r"E:\Datasets\food-101\meta\test.json"  
  weights_dir=r"E:\Datasets\food-101\weights\ResNet152"

  #하이퍼 파라미터 
  save_point=20 #20 에폭마다 가중치 저장
  batch=32  
  workers=8 #data loader의 worker 수
  epochs=300  
  patience=0  #0이면 early stop 안함
  inplace=(227,227)

  transforms={  #케이스 별 transform 정의
      'train':torchvision.transforms.Compose([torchvision.transforms.Resize(inplace), torchvision.transforms.ToTensor()]),
      'valid':torchvision.transforms.Compose([torchvision.transforms.Resize(inplace), torchvision.transforms.ToTensor()]),
      'test':torchvision.transforms.Compose([torchvision.transforms.Resize(inplace), torchvision.transforms.ToTensor()])
      }
  
  #데이터 셋 및 데이터 로더 생성
  train_dataset=a_new_test.FoodSpottingDataset(img_dir,train_list,transforms['train'], 0,0.7,1)
  train_loader=data.DataLoader(train_dataset, batch, pin_memory=True, num_workers=workers, drop_last=False)
  valid_dataset=a_new_test.FoodSpottingDataset(img_dir,train_list,transforms['valid'],0.7,1,1)
  valid_loader=data.DataLoader(valid_dataset, batch, pin_memory=True, num_workers=workers, drop_last=False)  

  model=models.resnet152(weights=ResNet152_Weights.DEFAULT) #pretrained weight사용
  classes_num=len(train_dataset.label_map)  #데이터 셋의 클래스 개수 가져오기
  model.fc=nn.Linear(2048,classes_num)  #모델의 출력을 101로 설정

  criterion=nn.CrossEntropyLoss(reduction='sum')  #손실 함수에서, 모든 손실을 더함
  optimizer=torch.optim.Adam(model.parameters())  #옵티마이저 설정

  a_new_test.train(model,epochs,save_point,patience,  #훈련 시작
                weights_dir,train_loader,valid_loader,criterion,optimizer)

if __name__=='__main__':
  os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #numpy와 충돌을 방지하기 위해 설정
  main()
