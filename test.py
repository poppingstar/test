import random, os, time, collections
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
import torch, torchvision
import torch.nn as nn
from torchvision.models import alexnet,AlexNet_Weights, resnet152,ResNet152_Weights,efficientnet_b6,EfficientNet_B6_Weights,inception_v3,Inception_V3_Weights
import numpy as np

class FoodSpottingDataset(torch.utils.data.Dataset):  #데이터 셋 클래스 선언
  def __init__(self,image_dir,labels_file,transform,start=0,end=1):
    self.image_dir=image_dir  #이미지 폴더 경로
    self.transform=transform  #transform 객체
    self.label_map={}  #클래스 이름을 숫자로 매핑

    with open(labels_file,'r') as f:		#레이블 목록 파일에 읽기 모드로 접근
      d=collections.defaultdict(list) #레이블별 파일 목록을 임시로 저장하는 딕셔너리
      for line in f:	#라인별로 읽어옴
        label, image_id =line.strip().split('/')	#좌우 공백을 없애고(있다면) /를 기준으로 문자열 분할, 각각은 레이블과 이미지 아이디를 나타냄
        if label not in self.label_map:	#레이블이 기존에 매핑되어있지 않다면
          self.label_map[label]=len(self.label_map)  #매핑 테이블에 고유 클래스 번호 할당
        d[self.label_map[label]].append(image_id) #각 레이블 번호에 일치하는 이미지 파일명 리스트를 매핑

      self.labels=[(item, i)
                  for i in d 
                  for item in d[i][int(len(d[i])*start):int(len(d[i])*end)]]  #이미지 ID와 클래스 번호를 저장

  def __len__(self):	#인스턴스를 len의 인자로 인스턴스를 전달할 떄 
    return len(self.labels)	#레이블 배열의 길이 반환

  def __getitem__(self, idx):	#이터러블로 정의
    image_id, label = self.labels[idx]	#레이블의 인덱스를 순회하며 이미지 ID와 클래스 번호 가져옴
    # 이미지 경로 생성
    image_path = os.path.join(self.image_dir, list(self.label_map.keys())[label], f"{image_id}.jpg")	#이미지/레이블/개별_이미지에 접근
    image = Image.open(image_path).convert("RGB")	#이미지를 읽어오며 RGB로 변환
    
    if self.transform: #transform이 있다면
      image = self.transform(image) #이미지에 적용한다
    
    return image, torch.tensor(label) #이미지와 텐서로 복사된 레이블을 반환

@dataclass  #메타 데이터를 저장하는 클래스
class Hyper():
  name:str  #모델의 이름
  inplace:int #모델의 입력 크기
  resize:int #축소된 사진의 크기(짧은 쪽)
  batch:int #모델의 배치 사이즈

class Trainer():  #네트워크 및 훈련을 위한 객체
  def __init__(self,model:nn.Module,loss_func,dclass:Hyper):
    self.model=model  #모델 객체
    self.loss_func=loss_func  #손실 함수
    self.hyper=dclass  #메타 데이터 저장 클래스
    self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #사용 가능한 gpu가 있으면 gpu를 디바이스로 설정
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #PCI Bus 순으로  GPU를 나열
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #0번 gpu 사용
  
  def set_optimizer(self, optimizer:torch.optim.Optimizer, **kwargs): #옵티마이저 설정
    self.optimizer=optimizer(self.model.parameters(), **kwargs) 
  
  def set_scheduler(self,schedular:torch.optim.lr_scheduler,**kwargs):  #스케줄러 설정
    self.scheduler=schedular(self.optimizer,**kwargs)

  def calc_accuracy(self,pred,label): #accuracy 계산
    _,top1=torch.max(pred,1)
    correct=(top1==label).sum().item()
    acc=correct/label.size(0)
    return acc*100

  def _fit(self): #에폭 하나를 훈련하는 함수
    self.model.train()
    epoch_loss=.0
    for imgs, labels in self.train_loader:
      imgs=imgs.to(self.device); labels=labels.to(self.device)      
      self.optimizer.zero_grad()

      outputs=self.model(imgs)
      loss=self.loss_func(outputs, labels)
      loss.backward()
      self.optimizer.step()
      epoch_loss+=loss.item()
    epoch_loss/=len(self.train_loader.dataset) #평균 손실 계산
    
    return epoch_loss
    
  def _valid(self): #검증 함수
    with torch.inference_mode():
      self.model.eval()
      epoch_acc,epoch_loss=.0,.0

      for imgs, labels in self.valid_loader:
        imgs=imgs.to(self.device); labels=labels.to(self.device)
        
        outputs=self.model(imgs)
        loss=self.loss_func(outputs, labels)
        epoch_loss+=loss.item()
        acc=self.calc_accuracy(outputs,labels) 
        epoch_acc+=acc*imgs.size(0)

    epoch_loss/=len(self.valid_loader.dataset) #평균 손실 계산
    epoch_acc/=len(self.train_loader.dataset)
    return epoch_loss, epoch_acc

  def train(self, total_epochs, save_dir, patience=0, save_point=50): #훈련 함수
    print(self.device)  #학습에 사용되는 장치를 확인
    minimum_loss,early_stop_count,total_time=100,0,0
    self.model=self.model.to(self.device)
    graph=ValidGraph(total_epochs)  #Valdiation 그래프 객체 생성

    for epoch in range(total_epochs): #정해진 에폭 횟수만큼 실행
      st=time.time() 
      train_loss=self._fit()

      #valid loss 계산
      val_loss, val_acc=self._valid()
      graph.update(epoch,val_loss,val_acc)  #그래프 업데이트
      
      if epoch%save_point==0: #세이브 포인트에 도달하면 저장
        self.save_param(save_dir,f'weights_{epoch}.pth',True)
        graph.save(save_dir,'graph.png',True)
        # torch.save(self.model.state_dict(), os.path.join(save_dir, f'weights_{epoch}.pth'))
      
      #early stopping
      if minimum_loss<=val_loss and patience>0:
        if (early_stop_count:=early_stop_count+1)>=patience:
          print('early stop')
          graph.save(save_dir,'graph.png',True)
          break
      else: #최고 weight 갱신
        # torch.save(self.model.state_dict(), best_file)
        self.save_param(save_dir,'weight_best.pth',True)  #weight파일 저장
        early_stop_count=0 
        minimum_loss=val_loss
      duration=time.time()-st
      total_time+=duration  #한 에폭의 소요 시간 계산
      
      print(f'Epoch{epoch+1}/{total_epochs}, Train Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f},  Minimun Loss: {minimum_loss:.2f},'
             f'Validation Accuracy: {val_acc:.2f}, Duration: {duration:,.0f}, total_time: {total_time:,.0f}')  #에폭마다 결과 출력

  def test(self): #테스트 함수
    self.model.eval()
    total, top1_correct, top5_correct=0,0,0
    self.model=self.model.to(self.device)

    with torch.no_grad():
      for imgs, labels in self.test_loader:
        imgs=imgs.to(self.device)
        labels=labels.to(self.device)

        outputs=self.model(imgs)
        _, top5_pred=torch.topk(outputs.detach(), 5, dim=1, largest=True, sorted=True)  #가장 확률이 높은 이미지 5개를 내림차순으로 반환
        total+=labels.size(0) #지금까지 추론한 이미지 갯수를 더함
        
        top1_pred=top5_pred[:,0]  #top1 레이블만 뽑아냄
        top1_correct+=(top1_pred==labels).sum().item()  #top1이 정확한 경우 누적
        top5_correct += torch.sum(torch.sum(top5_pred == labels.data.view(-1, 1), dim=1) > 0) #top5가 정확히 예측한 경우 누적
        # top5_correct+= ([torch.any(labels==row) for row in top5_predicted_classes]).sum().item() 

    top1_accuracy=top1_correct/total 
    top5_accuracy=top5_correct/total
    
    return top1_accuracy,top5_accuracy  
  
  def set_data_loaders(self,datasets):  #각각의 데이터 로더 설정
    self.train_loader=torch.utils.data.DataLoader(datasets[0], self.hyper.batch, shuffle=False, pin_memory=True, num_workers=0, drop_last=True)
    self.valid_loader=torch.utils.data.DataLoader(datasets[1], self.hyper.batch, shuffle=False, pin_memory=True, num_workers=0, drop_last=True)
    self.test_loader=torch.utils.data.DataLoader(datasets[2], self.hyper.batch, shuffle=False, pin_memory=True, num_workers=0, drop_last=True)

  def get_hyper(self):
    return self.hyper #메타 데이터 클래스 반환
  
  def save_param(self,dir,file_name, replace=False):  #파라미터 저장
    name,ext=os.path.splitext(file_name)
    ext=ext if ext else '.pth'
    file_name=name+ext
    save_path=os.path.join(dir,file_name)

    if not replace:
      save_path=no_overwrite(save_path) #기존 파일이 덮어써지는 것을 방지

    torch.save(self.model.state_dict(),save_path)

class ValidGraph: #그래프 클래스
  def __init__(self,epoch_lim,loss_lim=20,xticks=10,yticks=10,acc_min=0):
    self.x:list=[]  #에폭을 저장 리스트
    self.loss:list=[] #loss를 저장할 리스트
    self.acc:list=[] #Accuray를 저장할 리스트
    self._make(epoch_lim,acc_min,loss_lim,xticks,yticks)  #그래프 설정 및 생성
     
  def _preprocessing(self): #한글 깨짐 방지
    mpl.rcParams['font.family']='Malgun Gothic'
    mpl.rcParams['font.size']=15
    mpl.rcParams['axes.unicode_minus']=False

  def _make(self,epoch_lim,acc_lim,loss_min,xticks,yticks):
    self._preprocessing()
    _, self.ax1 = plt.subplots()
    
    self.ax1.set_xlabel('Epochs')
    self.ax1.set_xlim(0,epoch_lim)
    self.ax1.set_xticks(range(0,epoch_lim,epoch_lim//xticks))
    self.ax1.plot(self.x, self.loss, 'b-', label="Loss")
    self.ax1.set_ylabel('Loss', color='b')
    self.ax1.tick_params(axis='y', labelcolor='b')
    self.ax1.set_ylim(0,loss_min)
    self.ax1.set_yticks(range(0,loss_min,loss_min//yticks))
    
    self.ax2 = self.ax1.twinx()  
    self.ax2.plot(self.x, self.acc, 'r--', label='Accuracy')
    self.ax2.set_ylabel('Accuracy', color='r')
    self.ax2.tick_params(axis='y', labelcolor='r')
    self.ax2.set_ylim(acc_lim,100)
    self.ax2.set_yticks(range(acc_lim,100,(100-acc_lim)//yticks))

    plt.title('Loss-Accuracy Graph')
    plt.draw()
    plt.pause(.1)

  def update(self,epochs,loss,acc): #데이터를 업데이트
    self.x.append(epochs); self.loss.append(loss); self.acc.append(acc)
    self.ax1.plot(self.x,self.loss, 'b-', label="Loss")
    self.ax2.plot(self.x,self.acc,'r--', label='Accuracy')

    plt.draw()  #그래프 다시 그리기
    plt.pause(1)
  
  def save(self,dir,file_name, replace=False):  #그래프 저장
    name,ext=os.path.splitext(file_name)
    ext= ext if ext else '.png'
    file_name=name+ext
    save_path=os.path.join(dir,file_name)

    if not replace: #replace 옵션이 없다면
      save_path=no_overwrite(save_path) #파일을 덮어쓰는것을 방지
    plt.savefig(save_path)

def create_transfrom(imagenet_norm=True,*args): #개별 transform을 생성하는 함수
  transforms=[*args]  

  if not any(isinstance(transform, torchvision.transforms.ToTensor) for transform in transforms): #To Tensor 클래스가 없다면 추가
    transforms.append(torchvision.transforms.ToTensor())

  if imagenet_norm: #이미지넷 기반의 정규화 적용
    imagenet_norm=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.append(imagenet_norm)

  return torchvision.transforms.Compose(transforms) #데이터 전처리를 일괄 처리

def create_transforms(imagenet_norm=True,*,train,valid,test): #transform을 한번에 생성 하는 함수
  train_transform=create_transfrom(imagenet_norm,*train)  
  valid_transform=create_transfrom(imagenet_norm,*valid)
  test_transform=create_transfrom(imagenet_norm,*test)

  transforms=collections.namedtuple('datasets', 'train valid test')(train_transform,valid_transform,test_transform) #각 데이터 셋을 네임드 튜플에 담아 반환, 그냥 네임드 튜플 써보고 싶었어요

  return transforms
  
def create_datasets(img_dir, list_files, transforms, split=0.3):  #데이터셋 생성
  idx=0
  # train_set=FoodSpottingDataset(img_dir, list_files[idx], transforms[idx]); idx+=1

  if split:
    train_set=FoodSpottingDataset(img_dir, list_files[idx], transforms[idx],end=1-split); idx+=1
    valid_set=FoodSpottingDataset(img_dir, list_files[idx], transforms[idx],start=1-split); idx+=1
    # valid_len=int(len(train_set)*split)
    # train_len=len(train_set)-valid_len
    # train_set, valid_set=torch.utils.data.random_split(train_set,[train_len,valid_len]); idx+=1 
  else:
    valid_set=FoodSpottingDataset(img_dir, list_files[idx], transforms[idx]); idx+=1
  
  test_set=FoodSpottingDataset(img_dir, list_files[idx], transforms[idx])
  datasets=collections.namedtuple('DataSets', 'train valid test')(train_set,valid_set,test_set)  
  
  return datasets

class InceptionTrainer(Trainer):  #인셉션 넷은 aux_logit의 존재로 훈련 함수를 재정의 했습니다
  def _fit(self):
    self.model.train()
    epoch_loss=.0
    for imgs, labels in self.train_loader:
      imgs=imgs.to(self.device); labels=labels.to(self.device)      
      self.optimizer.zero_grad()

      outputs=self.model(imgs)
      outputs=outputs.logits+outputs.aux_logits*0.3 #aux logit이 30%의 영향을 미치도록 조정, 유일한 변경점이에요
      loss=self.loss_func(outputs, labels)
      loss.backward()
      self.optimizer.step()
      epoch_loss+=loss.item()
    epoch_loss/=len(self.train_loader.dataset) #평균 손실 계산
    
    return epoch_loss

class Pca():  #함수 객체로 수정, 애는 GPT로 생성했습니다
  def __call__(self,image,alpha_std=0.1):
    # 이미지 텐서가 ( C, H, W) 형태여야 하므로 이미지 텐서를 (C, H*W) 형태로 변환
    img_flat = image.view(3, -1)  # (C, H*W) 형태로 변환, C=3
    
    # 1. 공분산 행렬 계산
    cov_matrix = torch.cov(img_flat)  # (3, 3) 공분산 행렬 계산
    
    # 2. 고유값 및 고유벡터 계산
    eig_vals, eig_vecs = torch.linalg.eigh(cov_matrix)  # 고유값과 고유벡터 계산
    
    # 3. 임의의 alpha 생성 (표준 편차가 alpha_std인 가우시안 분포)
    alpha = torch.normal(0, alpha_std, size=(3,))  # 표준편차 alpha_std인 가우시안 노이즈
    
    # 4. 고유값에 비례하는 노이즈 생성
    noise = torch.matmul(eig_vecs, alpha * eig_vals)  # 고유벡터와 고유값을 결합하여 노이즈 생성
    
    # 5. 이미지에 노이즈 추가
    img_augmented_flat = img_flat + noise.view(-1, 1)  # 각 픽셀에 노이즈 추가 (C, H*W) 형태로 변경
    
    # 6. 결과 이미지를 원래 크기(3, H, W)로 복원하고, [0, 1] 범위로 클리핑
    img_augmented = torch.clamp(img_augmented_flat.view(3, *image.shape[1:]), 0, 1)  # (C, H, W)로 복원
      
    return img_augmented

def no_overwrite(path): #기존 훈련 파일이 덮어써지지 않도록 하는 함수
  dir_path=os.path.dirname(path)
  file_name=os.path.basename(path)
  name,ext=os.path.splitext(file_name)
  i=1
  while os.path.exists(path):
    file_name=f'{name}_{i}{ext}'
    path=os.path.join(dir_path,file_name)
    i+=1
  return path

if __name__=='__main__':  #메인 함수
  def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #경로 설정
    img_dir=r'E:\Datasets\food-101\images'
    train_list=r"E:\Datasets\food-101\meta\train.txt"
    test_list=r"E:\Datasets\food-101\meta\test.txt"  
    weights_dir=r"E:\Datasets\food-101\weights"

    #하이퍼 파라미터 
    save_point=50
    batch=64
    epochs=300
    patience=20

    loss_func=nn.CrossEntropyLoss(reduction='sum')  #손실 함수
    
    #개별 모델 객체 생성, 가중치는 ImageNET기준
    alexNet=alexnet(weights=AlexNet_Weights.DEFAULT)  
    resNet152=resnet152(weights=ResNet152_Weights.DEFAULT)
    inceptionNet=inception_v3(weights=Inception_V3_Weights.DEFAULT,)
    efficientNet_b6=efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)

    #각 모델의 메타정보를 저장할 데이터 클래스
    alexnet_meta=Hyper('AlexNet',227,256,batch)
    resnet_meta=Hyper('ResNet_152',227,256,batch)
    inception_meta=Hyper('InceptionV3',299,299,batch)
    efficient_meta=Hyper('EfficientNet',227,256,batch)

    #훈련할 모델 튜플
    models2train=(alexNet:=Trainer(alexNet,loss_func,alexnet_meta),
                  # resNet152:=Trainer(resNet152, loss_func,resnet_meta),
                  # inceptionNet:=InceptionTrainer(inceptionNet, loss_func,inception_meta),
                  # efficientNet_b6:=Trainer(efficientNet_b6,loss_func,efficient_meta),
                  )

    #모델을 순회하며 훈련 및 결과 저장
    for model in models2train:
      meta=model.get_hyper()

      #훈련마다 새로운 하위 디렉토리에 저장
      i=1
      while 1:
        save_dir=os.path.join(weights_dir,meta.name,f'train_{i}')
        if not os.path.isdir(save_dir):
          os.makedirs(save_dir,exist_ok=True)
          break
        i+=1

      #훈련 전처리 정의
      train=(torchvision.transforms.Resize(meta.resize),  
              torchvision.transforms.RandomResizedCrop(meta.inplace),
              torchvision.transforms.RandomHorizontalFlip(),
              torchvision.transforms.ToTensor(),
              Pca())
      #테스트 전처리 정의
      test=(torchvision.transforms.Resize(meta.resize), 
            torchvision.transforms.FiveCrop(meta.inplace),
            torchvision.transforms.Lambda(lambda crops: crops+[torchvision.transforms.RandomHorizontalFlip(1)(crop)for crop in crops]),
            torchvision.transforms.ToTensor())

      torch.cuda.empty_cache()  #GPU 메모리 정리
      model.set_optimizer(torch.optim.Adam)  #옵티마이저 설정
      transforms=create_transforms(train=train,valid=train,test=test) 
      datasets=create_datasets(img_dir,(train_list,train_list,test_list),transforms) #각 데이터 셋 생성
      model.set_data_loaders(datasets)    #데이터로더 설정

      print(meta.name)  #어떤 모델이 돌아가는지 확인하기 위해 출력
      model.train(epochs,save_dir,patience,save_point) 
      model.save_param(save_dir,'last_weight.pth')  #종료 시점 파라미터 저장

      start=time.time() #시간 측정. 나중에 데코레이터로 바꾸까 싶어요
      top1,top5=model.test()  #테스트 결과인 accuracy 저장
      end=time.time()
      inference_latency=end-start 

      #테스트 결과를 txt 파일로 저장
      results_file=os.path.join(save_dir,'results.txt') 
      with open(results_file,'w') as f: 
        s=f'''top1 accuracy: {top1}
        top5 accuracy: {top5}
        inference latency={inference_latency}
        '''
        f.write(s)

  main()  #메인 함수 실행
