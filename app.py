from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import os
import zipfile
import csv
import datetime
import torch
import torch.backends.cudnn as cudnn
import torchaudio
import torchaudio.transforms as transforms
import cv2
import numpy as np
import math
import imquality.brisque as brisque
import PIL.Image

from FaceDetector.utils.helpers import decode_output, do_nms
from FaceDetector.models.retinaface import RetinaFace
from ultralytics import YOLO

from utils.deid import GaussianBlur, Masking, DataReplace, PitchShift, AddNoise, Resample

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database_file.db'  # 데이터베이스 URI 설정
db = SQLAlchemy(app)  # db 인스턴스 생성 및 초기화

# LoginManager 설정
login_manager = LoginManager()
login_manager.login_view = '/login'  # 로그인 페이지의 라우트 이름을 지정
login_manager.init_app(app)

# 로깅 설정
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

UPLOAD_FOLDER = os.path.join(app.static_folder, 'media_files')
rgb_mean = (104, 117, 123)

# 월별 평균 일출 및 일몰 시간을 시간 단위로 나타낸 데이터 (임의의 데이터)
average_sunrise_times = [7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
average_sunset_times = [17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 19.5, 19.0, 18.5, 18.0, 17.5]

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    consent = db.Column(db.String(20))
    risk_level = db.Column(db.String(20))
    file_size = db.Column(db.Float, nullable=False)

class Anonymizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_model = None
        self.lp_model = None
        
    def load_model(self):
        if self.face_model or self.lp_model is None:
            print(f'Using {self.device}')
            self.face_model = RetinaFace(phase='test')
            self.ckpt = torch.load('weights/resnet50_ckpt-best.pth')
            self.face_model.load_state_dict(self.ckpt['network'])
            self.face_model.to(self.device)
            self.face_model.eval()

            self.lp_model = YOLO('weights/license_plate_detector.pt')

            if self.device == 'cuda':
                cudnn.benchmark = True
    
    def detector(self, consent, img_raw):
        image_anonymization_method = request.form.get('image_anonymization_method')
        faces_num = 0
        lp_num = 0
        
        img = img_raw.astype(np.float32)
        img -= rgb_mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        img = img.to(self.device)
        img = torch.unsqueeze(img, dim=0)

        _, out = self.face_model(img)
        loc, conf, landms = out
        scores, boxes, landms = decode_output(img, loc, conf, landms, self.device)
        face_dets = do_nms(scores, boxes, landms, CONFIDENCE_THRESHOLD=0.5, NMS_THRESHOLD=0.45)

        lp_dets = self.lp_model(img_raw)[0]

        max_face = []
        max_lp = []

        for face_det in face_dets:
            x1_f, y1_f, x2_f, y2_f = face_det[0], face_det[1], face_det[2], face_det[3]
            # bpoints = list(map(int, bpoints))
            img_copy = img_raw.copy()
            face_roi = img_copy[int(y1_f):int(y2_f), int(x1_f):int(x2_f)]

            if not face_roi.size:
                continue
            
            face_height = int(y2_f - y1_f)
            max_face.append(face_height)

            if consent == 'on':
                radius_x = int((x2_f - x1_f) / 2)
                radius_y = int((y2_f - y1_f) / 2)

                mask = np.zeros_like(face_roi)
                cv2.ellipse(mask, (radius_x, radius_y), (radius_x, radius_y), 0, 0, 360, (255, 255, 255), -1)

                if image_anonymization_method == 'blur':
                    face_roi = GaussianBlur(face_roi)
                elif image_anonymization_method == 'mask':
                    face_roi = Masking(face_roi)
                elif image_anonymization_method == 'replace':
                    face_roi = DataReplace(face_roi, 'utils/face.png', 'utils/face_mask.png')
                    
                face_roi = cv2.bitwise_and(face_roi, mask)

                img_raw[int(y1_f):int(y2_f), int(x1_f):int(x2_f)] = cv2.bitwise_and(img_raw[int(y1_f):int(y2_f), int(x1_f):int(x2_f)], cv2.bitwise_not(mask))
                img_raw[int(y1_f):int(y2_f), int(x1_f):int(x2_f)] = cv2.add(img_raw[int(y1_f):int(y2_f), int(x1_f):int(x2_f)], face_roi)

            elif consent == None:
                pass

            faces_num += 1

        for lp_det in lp_dets.boxes.data.tolist():
            x1_lp, y1_lp, x2_lp, y2_lp, _, _ = lp_det

            img_copy = img_raw.copy()
            lp_roi = img_copy[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp), :]

            if not face_roi.size:
                continue

            lp_height = int(y2_lp - y1_lp)
            max_lp.append(lp_height)

            if consent == 'on':                
                if image_anonymization_method == 'blur':
                    lp_roi = GaussianBlur(lp_roi)
                elif image_anonymization_method == 'mask':
                    lp_roi = Masking(lp_roi)
                elif image_anonymization_method == 'replace':
                    lp_roi = DataReplace(lp_roi, 'utils/lp.png', 'utils/lp_mask.png')

                img_raw[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp), :] = lp_roi

            elif consent == None:
                pass

            lp_num += 1
            
        if len(max_face) > 0:
            face_height = max(max_face)
        else:
            face_height = None

        if len(max_lp) > 0:
            lp_height = max(max_lp)
        else:
            lp_height = None

        return img_raw, faces_num, face_height, lp_num, lp_height

    def risk_weight(self, input_img, month, time, weather):
        print(input_img.shape)
        sunrise_time = average_sunrise_times[month - 1]
        sunset_time = average_sunset_times[month - 1]
        midpoint_time = (sunset_time + sunrise_time) / 2
        
        if sunrise_time - 1 <= time < midpoint_time:
            brightness = 1.0 - 0.4 * (1 - math.exp(-0.1 * ((midpoint_time - 1) - time)**2))
        elif midpoint_time <= time <= sunset_time + 1:
            brightness = 0.6 + 0.4 * (1 - math.exp(-0.5 * ((sunset_time + 1) - time)**2))
        else:
            # 밤 시간대 (일출 이전, 일몰 이후)
            brightness = 0.6

        brisque_score =  brisque.score(input_img)
        normalized_brisque_score = 1.0 - (brisque_score / 100) * 0.2

        print(normalized_brisque_score)

        risk_weight = brightness * weather

        return risk_weight

    def risk_evaluation(self, faces_num, face_height, lp_num, lp_height, risk_weight):

        identifier_num = faces_num + lp_num

        if identifier_num == 0:
            existence_risk = 0
        elif 1 <= identifier_num < 3:
            existence_risk =  1
        elif 3 <= identifier_num < 5:
            existence_risk =  2
        elif 5 <= identifier_num < 7:
            existence_risk =  3
        elif 7 <= identifier_num < 9:
            existence_risk =  4
        elif 9 <= identifier_num:
            existence_risk =  5

        if face_height is None:
            perception_risk_f = 0
        elif 10 <= face_height < 20:
            perception_risk_f = 1
        elif 20 <= face_height < 40:
            perception_risk_f = 5
        elif 40 <= face_height:
            perception_risk_f = 10

        if lp_height is None:
            perception_risk_lp = 0
        elif 10 <= lp_height < 15:
            perception_risk_lp = 1
        elif 15 <= lp_height < 30:
            perception_risk_lp = 5
        elif 30 <= lp_height:
            perception_risk_lp = 10

        risk_level = existence_risk * risk_weight * max(perception_risk_f, perception_risk_lp)
        risk_level = min(max(risk_level, 0), 25)
        print('위험도 점수 : ', risk_level)

        if risk_level == 0:
            return '없음'
        elif risk_level <= 5:
            return '매우 낮음'
        elif risk_level <= 10:
            return '낮음'
        elif risk_level <= 15:
            return '보통'
        elif risk_level <= 20:
            return '높음'
        elif risk_level <= 25:
            return '매우 높음'
        
    def process_uploaded_file(self, file):
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(file_path)

        month = int(request.form.get('month'))
        time = int(request.form.get('time'))
        weather = float(request.form.get('weather'))

        consent = request.form.get('consent')
        audio_anonymization_method = request.form.get('audio_anonymization_method')

        _, file_extension = os.path.splitext(file_path)
        self.load_model()  # 모델 로딩

        # 익명화 방법에 따라 처리
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            result, face_num, face_height, lp_num, lp_height = self.detector(consent, img)
            risk_weight = self.risk_weight(img, month, time, weather)
            risk_level = self.risk_evaluation(face_num, face_height, lp_num, lp_height, risk_weight)
            cv2.imwrite(file_path, result)

        elif file_extension.lower() in ['.mp4', '.avi', '.mov']:
            video_path = os.path.join(UPLOAD_FOLDER, secure_filename(os.path.basename(file.filename)))

            video_capture = cv2.VideoCapture(file_path)
            if not video_capture.isOpened():
                flash('동영상 파일을 열 수 없습니다.', 'error')
                return redirect(url_for('index'))  # 또는 에러 페이지로 리다이렉트

            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 코덱 사용
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                result, _, _, _, _= self.detector(consent, frame)
                out.write(result)

            video_capture.release()
            out.release()

            risk_level = '보통'
    
        elif file_extension.lower() in ['.mp3', '.wav']:
            audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(os.path.basename(file.filename)))

            if audio_anonymization_method == 'pitchshift':
                PitchShift(audio_path, semitones=3)
            elif audio_anonymization_method == 'addnoise':
                AddNoise(audio_path, noise_factor=0.5)
            elif audio_anonymization_method == 'resample':
                Resample(audio_path, sampling_ratio=1.2)
            
            risk_level = '보통'  # You can customize this message
    
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 파일 크기 계산 (MB 단위)

        return consent, risk_level, round(file_size, 3)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('signup_username')
        password = request.form.get('signup_password')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('이미 존재하는 사용자 이름입니다. 다른 이름을 선택하세요.', 'error')
        else:
            new_user = User(username=username, password=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            flash('회원 가입이 완료되었습니다. 로그인하세요.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html')

@login_manager.unauthorized_handler
def unauthorized():
    return redirect(url_for('login'))  # 로그인 페이지로 리다이렉트

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('login_username')
        password = request.form.get('login_password')

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            logging.info(f"User '{username}' logged in")
            return redirect(url_for('index'))
        else:
            flash('로그인에 실패했습니다. 아이디와 패스워드를 확인하세요.', 'error')
            logging.warning(f"Failed login attempt for user '{username}'")

    return render_template('login.html')

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, user_id)  # 문자열로 변환하지 않고 그대로 사용

@app.route('/logout')
@login_required
def logout():
    logout_user()  # 사용자 로그아웃 처리
    flash('로그아웃되었습니다.', 'success')
    return redirect(url_for('index'))

@app.route('/')
@login_required  # 로그인된 사용자만 접근 가능한 페이지
def index():
    uploaded_files = UploadedFile.query.all()

    return render_template('index.html', uploaded_files=uploaded_files)

@app.route('/upload', methods=['POST'])
def upload():
    anonymizer = Anonymizer()
    uploaded_filenames = []

    if 'file' in request.files:
        files = request.files.getlist('file')
        for file in files:
            filename = secure_filename(file.filename)
            consent, risk_level, file_size = anonymizer.process_uploaded_file(file)  # 수정된 부분

            uploaded_filenames.append(filename)

            uploaded_file = UploadedFile(filename=filename, consent=consent, risk_level=risk_level, file_size=file_size)  # 수정된 부분
            db.session.add(uploaded_file)
            db.session.commit()

    if uploaded_filenames and consent is None:
        flash('파일이 업로드되었습니다.')
    elif uploaded_filenames and consent == 'on':
        flash('익명화 요청이 확인되었습니다. 익명화된 파일이 업로드되었습니다.')

    return redirect(url_for('index'))

@app.route('/preview/<filename>')
def preview(filename):
    return send_from_directory('static/media_files', filename)

@app.route('/download_selected', methods=['POST'])
def download_selected():
    selected_files = request.form.getlist('selected_files')

    if not selected_files:
        flash('다운로드할 파일을 선택하세요.', 'error')
        return redirect(url_for('index'))

    zip_filename = 'selected_files.zip'
    zip_path = os.path.join(app.static_folder, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in selected_files:
            uploaded_file = UploadedFile.query.filter_by(filename=filename).first()
            if uploaded_file:
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                zipf.write(file_path, filename)

    return send_from_directory(app.static_folder, zip_filename)

@app.route('/delete_selected', methods=['POST'])
def delete_selected():
    selected_files = request.form.getlist('selected_files')

    def is_admin(user):
        return user.is_authenticated and user.username == 'admin'

    if not selected_files:
        flash('삭제할 파일을 선택하세요.', 'error')
        return redirect(url_for('index'))

    elif is_admin(current_user):  # Only admins can delete files
        success_messages = []
        error_messages = []
        for filename in selected_files:
            uploaded_file = UploadedFile.query.filter_by(filename=filename).first()
            if uploaded_file:
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                db.session.delete(uploaded_file)  # Delete the database record
                db.session.commit()
                success_messages.append(filename)
            else:
                error_messages.append(filename)

        if success_messages:
            flash(f'파일 {success_messages}이 삭제되었습니다.', 'success')
        if error_messages:
            flash(f'파일 {filename}을 찾을 수 없습니다.', 'error')

    else:
        flash('권한이 없습니다.', 'error')

    return redirect(url_for('index'))

@app.route('/change_password_page', methods=['GET'])
@login_required
def change_password_page():
    return render_template('change_password.html')

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    if not check_password_hash(current_user.password, current_password):
        flash('현재 비밀번호가 일치하지 않습니다.', 'error')
        return redirect(url_for('change_password_page'))

    if new_password != confirm_password:
        flash('새로운 비밀번호와 확인 비밀번호가 일치하지 않습니다.', 'error')
        return redirect(url_for('change_password_page'))

    current_user.password = generate_password_hash(new_password)
    db.session.commit()

    flash('비밀번호가 성공적으로 변경되었습니다. 다시 로그인해주세요.', 'success')
    logout_user()  # 로그아웃 처리
    return redirect(url_for('login'))  # 로그인 페이지로 리다이렉트

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
