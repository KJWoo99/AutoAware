## 설치해야할 라이브러리

### client 파일로 이동 후 설치

#### pip install npm

#### npm install react-dom react-router-dom leaflet react-leaflet lucide-react

### server 파일로 이동 후 설치

#### pip install PyJWT pymongo bcrypt fastapi uvicorn pythn_dotenv opencv-python aiortc python-multipart

<hr>

## 시작 방법

#### 터미널에 cd client 후 npm start or npm start dev

#### 또 다른 터미널에 cd server 후 uvicorn main:app --reload --port=8000

<hr>

### Component

#### TopList 는 Face, List, MyPage, Register 각 component에 담겨있음, Sidebar 는 Header에 담겨있음

<hr>

### .env파일

#### mongodb key, secret key 설정해줘야함
