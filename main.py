import os
import shutil
from datetime import datetime, timedelta
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import hashlib

from backend.database import engine, get_db
import backend.models as models
from model.inference import LeukoNetModel
from model.grad_cam import generate_grad_cam

# Create DB schema
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="LeukoNet API")

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ✅ Templates FIX
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
templates.env.cache = None  # IMPORTANT FIX

# Load Model
ai_model = None
try:
    ai_model = LeukoNetModel()
except Exception as e:
    print(f"Warning: Model not loaded yet. {e}")

# ---- Auth System ----
SECRET_KEY = "leukonet_super_secret_key_12345"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 240

def get_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(models.Doctor).filter(models.Doctor.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ---- Auth APIs ----
@app.post("/api/signup")
def signup(
    username: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(...),
    hospital: str = Form(None),
    db: Session = Depends(get_db)
):
    if db.query(models.Doctor).filter(models.Doctor.username == username).first():
        raise HTTPException(400, "Username already exists")

    user = models.Doctor(
        username=username,
        password_hash=get_password_hash(password),
        full_name=full_name,
        hospital=hospital
    )
    db.add(user)
    db.commit()

    return {"success": True}

@app.post("/api/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(models.Doctor).filter(models.Doctor.username == username).first()

    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer",
        "full_name": user.full_name
    }

# ---- UI Pages (FIXED) ----
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/history")
def history_page(request: Request):
    return templates.TemplateResponse(request, "history.html", {})

@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})

# ---- Prediction ----
@app.post("/api/predict")
async def predict(
    patient_id: str = Form(...),
    patient_name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_doctor: models.Doctor = Depends(get_current_user)
):
    global ai_model

    if ai_model is None:
        ai_model = LeukoNetModel()

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload image only")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{patient_id}_{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ai_model.predict(file_path)

    heatmap_file = f"heatmap_{filename}"
    heatmap_path = os.path.join(UPLOAD_DIR, heatmap_file)

    if result["prediction"] != "Uncertain":
        generate_grad_cam(file_path, ai_model.model, heatmap_path)

    db_record = models.PatientRecord(
        patient_id=patient_id,
        patient_name=patient_name,
        image_path=f"/static/uploads/{filename}",
        heatmap_path=f"/static/uploads/{heatmap_file}",
        prediction=result["prediction"],
        confidence=result["confidence"]
    )
    db.add(db_record)
    db.commit()

    return {
        "success": True,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "distribution": result.get("distribution", {}),
        "metrics": result.get("metrics", {}),
        "image_url": f"/static/uploads/{filename}",
        "heatmap_url": f"/static/uploads/{heatmap_file}"
    }

# ---- History ----
@app.get("/api/history")
def history(
    db: Session = Depends(get_db),
    current_user: models.Doctor = Depends(get_current_user)
):
    records = db.query(models.PatientRecord).all()
    return {"records": records}