import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _WeightedLoss
import torchvision
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import time, os, json
import numpy as np
from pathlib import Path

class FoodSpottingDataset(torch.utils.data.Dataset):  #데이터 셋 클래스 선언
  def __init__(self,img_dir,labels_file,transform,start, end, raito=False):
    self.transform=transform  #transform 객체
    self.img_dir=Path(img_dir)
    self.label_map, self.data_set=self.make_dataset(labels_file,start,end, raito)
    
  def make_dataset(self, labels_path, start, end, raito):
    label_map={}
    data_lists=[]
    
    with open(labels_path) as f:
      dict_data=json.load(f)


      for key in dict_data:
        label_map[key]=len(label_map)
        
        if raito:
          element_len=len(dict_data[key])
          start_e=int(start*element_len)
          end_e=int(end*element_len)

        sliced_list=dict_data[key][start_e:end_e]
        for s in sliced_list:
          label,id=s.split('/')
          data_lists.append((label,id))

    return label_map, data_lists
    
  def __len__(self):	#인스턴스를 len의 인자로 인스턴스를 전달할 떄 
    return len(self.data_set)	#데이터셋 배열의 길이 반환

  def __getitem__(self, idx):	#이터러블로 정의
    label, image_id = self.data_set[idx]	#레이블의 인덱스를 순회하며 이미지 ID와 클래스 번호 가져옴

    # 이미지 경로 생성
    image_path = self.img_dir/label/f"{image_id}.jpg"	#이미지/레이블/개별_이미지에 접근
    image = Image.open(image_path).convert("RGB")	#이미지를 읽어오며 RGB로 변환
    
    if self.transform: #transform이 있다면
      image = self.transform(image) #이미지에 적용한다

    return image, torch.tensor(self.label_map[label], dtype=torch.long) #이미지와 텐서로 복사된 레이블을 반환

def run_epoch(model:nn.Module, loader:DataLoader, criterion:_WeightedLoss, optimizer:optim.Optimizer, device:torch.device, mode:str): #에폭 하나를 실행
  # torch.set_printoptions(profile='full',linewidth=200)  
  epoch_loss,epoch_acc=.0,.0  #loss, accuracy 초기화

  match mode:
    case 'train':  #훈련 모드시 모델을 훈련 모드로, gradient를 계산
      model.train()
      grad_mode=torch.enable_grad()
    case 'valid':  #validation모드에선 모델을 추론 모드로, gradient 계산 안함
      model.eval()
      grad_mode=torch.no_grad()
    case 'test':  #테스트모드에선 inference모드로
      model.eval()
      grad_mode=torch.inference_mode()  #no_grad보다 훨씬 강력한 모드
    case _:  #기타 케이스가 들어오면 에러 발생
      raise ValueError(f'Invalid mode {mode}') 
      
  for imgs, labels in loader:  #데이터 로더로부터 이미지와 레이블을 읽어오며
    imgs=imgs.to(device); labels=labels.to(device)  #학습 디바이스로 이동
    optimizer.zero_grad()  #옵티마이저의 그래디언트 초기화

    with grad_mode:  #각 모드 하에서 실행
      outputs=model(imgs)  #추론하고
      _,preds=torch.max(outputs,1)  #top 1예측값을 가져옴
      loss=criterion(outputs,labels)  #loss 계산
      
      if mode=='train':  #훈련 모드에선 역전파 포함
        loss.backward()
        optimizer.step()
      # for p in model.parameters():
      #   print(p.grad.norm())
    torch.cuda.empty_cache() #매 에폭마다 gpu메모리 정리
    epoch_loss+=loss.item()  #epoch loss에 batch별 loss 가산
    epoch_acc+=torch.sum(preds==labels)  #accuracy도 동일

  #criterion이 mean이면 부분 평균의 평균을 구하면 되겠지만, 마지막 batch의 사이즈가 달라질 수 있으므로 sum으로 변환한 후 데이터셋의 크기로 나눴습니다
  epoch_loss/=len(loader.dataset)  #epoch 평균 구하기
  epoch_acc/=len(loader.dataset)  #epoch 평균 구하기
  return epoch_loss, epoch_acc.item()  #loss와 acc의 평균 반환

def train(model:nn.Module,num_epochs:int,save_point:int,patience:int, save_dir:'str',  #훈련 함수
          train_loader, valid_loader, loss_func, optimizer):  
  device:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #사용 가능한 디바이스 확인
  print(device)
  model.to('cuda' if torch.cuda.is_available() else 'cpu')  #모델을 디바이스로 이동
  es_count, total_duration=0,0  #early stop 카운트와 총 수행시간을 0으로 초기화
  minimun_loss=float('inf')  #최소 loss를 무한대로 초기화
  save_dir=no_overwrite(save_dir)  #이전 학습 정보를 덮어쓰는것을 방지
  best_path=os.path.join(save_dir,'best_weight.pt')  #best weight 경로 설정
  last_path=os.path.join(save_dir,'last_weight.pt')  #last weight 경로 설정

  graph=LAGraph(num_epochs)  #그래프 객체 생성, 

  for epoch in range(num_epochs):
    since=time.time()  #에폭 시작 시간
    train_loss,train_accuracy=run_epoch(model, train_loader, loss_func, optimizer, device, 'train')  #훈련 실행
    valid_loss,valid_accuracy=run_epoch(model, valid_loader, loss_func, optimizer, device, 'valid')  #검증 실행
  
    graph.update(epoch,valid_loss, valid_accuracy)  #검증 결과로 그래프 업데이트
    graph.save(save_dir,'valid.png',True)  #그래프 저장
    
  #early stop
    if minimun_loss<=valid_loss:  #검증 로스가 최소치보다 크면
      es_count+=1  #es count를 증가시킨다
      if patience>0 and es_count>=patience:  #만약 patience가 0보다 크고, es_count가 patience보다 높다면
        torch.save(model.state_dict(),last_path)  #최종 훈련 가중치를 저장하고 학습 종료
        print('early stop')
        break

    else:  #현재 loss가 최소치면
      minimun_loss=valid_loss  #minimun loss를 갱신
      es_count=0  #early stop count를 초기화
      torch.save(model.state_dict(), best_path)  #best weight를 저장

  #save_point
    if epoch%save_point==0:  #세이브 포인트마다 에폭 저장
      save_point_path=os.path.join(save_dir,f'{epoch}_weight.pt')  #해당 에폭을 파일 명으로 가중치 저장
      torch.save(model.state_dict(), save_point_path) 
  #시간 계산

    duration=time.time()-since  #에폭 수행시간 계산
    total_duration+=duration  #총 수행시간에 합산
    print(f'epochs: {epoch+1}/{num_epochs}, train loss: {train_loss:.2f}, val loss: {valid_loss:.2f}, train accuracy:{train_accuracy:.2f}, '  #
          f'val accuracy: {valid_accuracy:.2f}, duration: {duration:.2f}, total duration: {total_duration:.2f}')
  torch.save(model.state_dict(),last_path)

def no_overwrite(path, mode='dir'): #기존 훈련 파일이 덮어써지지 않도록 하는 함수
  match mode:  
    case 'file':  #파일 레벨의 덮어쓰기 방지
      dir_path=os.path.dirname(path)  #경로에서 디렉토리 부분만 가져옴
      base=os.path.basename(path) #경로의 마지막 요소 반환
      #생각해보니 위 두개는 그냥 split을 하면 되는데 어째서..?
      file_name,ext=os.path.splitext(base)  #파일명과 확장자 분리
      i=1 #파일명에 추가할 숫자
      while os.path.exists(path): #해당 파일이 존재 시 파일명에 숫자를 붙임
        base=f'{file_name}_{i}{ext}'  
        path=os.path.join(dir_path,base)
        i+=1
      return path #유니크 경로 반환

    case 'dir': #디렉토리 레벨의 덮어쓰기 방지
      i=1
      new_dir=os.path.join(path,str(i)) #새로운 디렉토리 경로 생성
      while os.path.isdir(new_dir): #만약 해당 디렉토리가 존재하면
        i+=1
        new_dir=os.path.join(path,str(i)) #없는 디렉토리가 나올 때 까지 숫자를 증가시키며 적용
      os.makedirs(new_dir,exist_ok=True)  #디렉토리 생성
      return new_dir
      
class LAGraph: #그래프 클래스
  def __init__(self,epoch_lim,loss_lim=20,xticks=10,yticks=10,acc_min=0):
    self.x:list=[]  #에폭을 저장 리스트
    self.loss:list=[] #loss를 저장할 리스트
    self.acc:list=[] #Accuray를 저장할 리스트
    self._make(epoch_lim,acc_min,loss_lim,xticks,yticks)  #그래프 설정 및 생성
     
  def _preprocessing(self): #한글 깨짐 방지
    mpl.rcParams['font.family']='Malgun Gothic'
    mpl.rcParams['font.size']=15
    mpl.rcParams['axes.unicode_minus']=False

  def _make(self,epoch_lim,acc_lim,loss_min,xticks,yticks): #그래프 생성
    plt.ion()
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
    plt.pause(1)

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

  def __del__(self):
    plt.close()

def imshow(inp, title=None):
    """tensor를 입력받아 일반적인 이미지로 출력."""
    inp=inp.numpy().transpose(1,2,0)
    inp_255=inp*255 
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.
