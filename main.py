from fastapi import FastAPI, Request, Form, UploadFile,File, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from PIL import Image
from fastapi.responses import StreamingResponse
import io
import os
import models
import models1
from database import engine, sessionLocal
from sqlalchemy.orm import Session
from database import sessionLocal, engine
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from datetime import datetime

models.Base.metadata.create_all(bind=engine)
models1.Base.metadata.create_all(bind=engine)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 추론기
module = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
module.prepare(ctx_id=0, det_size=(640, 640))

# 데이터베이스 모델 생성
def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create a directory to store captured images
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# Create a directory to serve static files
statics_path = os.path.join("templates", "statics", "images")
if not os.path.exists(statics_path):
    os.makedirs(statics_path)

async def capture_and_save_image(image: UploadFile, member_name:str, member_phone: int ):
    # Save the captured image
    image_path = os.path.join(statics_path, "captured_image.jpg")
    
    with open(image_path, "wb") as f:
        f.write(await image.read())  # Use await to read the uploaded file content

    return image_path


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/sign", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("sign.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = Form(...) , name: str = Form(...), phone: str = Form(...), db: Session = Depends(get_db)):

    contents = await file.read()
    # Save the uploaded image
    with open(f"captured_images/{'test.jpg'}", "wb") as f:
        f.write(contents)

    if name or file or phone:
        return {'result' : '회원가입에 실패하셨습니다.'}
    
    isinstance = models.Member(member_name = name, member_phone = phone, member_image = contents)

    db.add(isinstance)
    db.commit()
    
    return {'result' : '회원가입에 성공하셨습니다.'}

@app.post("/login")
async def loain(file: UploadFile = Form(...), db: Session = Depends(get_db)):

    # 로그인시에 받아오는 사진
    content = await file.read()  # 파일 내용을 읽습니다.
    buffer = io.BytesIO(content) # 파일 내용을 바이트 스트림으로 변환합니다.
    image = Image.open(buffer)   # PIL을 사용하여 이미지로 열어옵니다.

    nparr = np.asarray(image)    # 이미지를 NumPy 배열로 변환합니다.
    nparr = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB)  # 이미지의 색상 형식을 변경합니다.

    faces = module.get(nparr)  
    feat = np.array(faces[0].normed_embedding, dtype=np.float32)

    # 저장된 이미지랑 db에서 조회해온 이미지 비교
    members = db.query(models.Member).all()

    for member in members:

        forImage = Image.open(io.BytesIO(member.member_image)) 
        nparr = np.asarray(forImage)  
        nparr = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB) 
        
        faces = module.get(nparr)
        featDB = np.array(faces[0].normed_embedding, dtype=np.float32) 

        sims = np.dot(feat, featDB)

        if sims> 0.5:
            isinstance = models1.Attendance(start = datetime.now())
            db.add(isinstance)
            db.commit()
            return f'로그인에 성공하셨습니다.{sims}'

    return f'로그인에 실패하셨습니다.{sims}'


# Serve static files
app.mount("/static", StaticFiles(directory=statics_path, html=True), name="static")