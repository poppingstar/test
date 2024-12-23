class FoodSpottingDataset(torch.utils.data.Dataset):  #데이터 셋 클래스 선언
  def __init__(self,image_dir,labels_file,transform,start=0,end=1):
    self.image_dir=image_dir  #이미지 폴더 경로
    self.transform=transform  #transform 객체
    self.label_map={}  #클래스 이름을 숫자로 매핑

    with open(labels_file,'r') as f:		#레이블 목록 파일에 읽기 모드로 접근
      d=collections.defaultdict(list) #레이블별 파일 목록을 임시로 저장하는 딕셔너리
      for line in f:	#라인별로 읽어옴
        label, image_id=line.strip().split('/')	#좌우 공백을 없애고(있다면) /를 기준으로 문자열 분할, 각각은 레이블과 이미지 아이디를 나타냄
        img_path=os.path.join(image_dir,label,f'{image_id}.jpg')
        if not os.path.exists(img_path):
          continue
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
    
    return image, torch.tensor(label, dtype=torch.long) #이미지와 텐서로 복사된 레이블을 반환
