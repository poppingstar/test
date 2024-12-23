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
