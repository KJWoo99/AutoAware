# RetinaFace 얼굴 검출 시스템

이 프로젝트는 RetinaFace 모델을 사용하여 이미지에서 얼굴을 검출하는 시스템입니다.

## 주요 기능

1. 이미지에서 얼굴 검출
2. 얼굴 경계 상자 (bounding box) 생성
3. 얼굴 랜드마크 검출
4. 신뢰도 기반 필터링
5. Non-Maximum Suppression (NMS) 적용

## 필요 라이브러리

- PyTorch
- NumPy
- OpenCV
- Matplotlib

## 사용 방법

1. 필요한 라이브러리를 설치합니다.
2. RetinaFace 모델 가중치 파일을 준비합니다.
3. 다음과 같이 `detection` 함수를 호출하여 얼굴을 검출합니다:

   ```python
   import cv2
   from face_detection import detection

   img = cv2.imread('sample_image.jpg')
   weights_path = 'path/to/model_weights.pth'
   detected_faces = detection(img, weights_path)

4. detected_faces는 검출된 얼굴의 경계 상자, 신뢰도 점수, 랜드마크 정보를 포함합니다.

# 주요 매개변수

confidence_threshold: 얼굴 검출 신뢰도 임계값 (기본값: 0.02)
nms_threshold: Non-Maximum Suppression 임계값 (기본값: 0.4)
top_k: NMS 전 유지할 최대 검출 수 (기본값: 5000)
keep_top_k: 최종적으로 유지할 최대 검출 수 (기본값: 750)

# 모델 정보
이 시스템은 MobileNet 기반의 RetinaFace 모델을 사용합니다. ResNet50 기반 모델로 전환하려면 cfg = cfg_re50로 설정을 변경하세요.

# 주의사항
GPU 사용이 가능한 경우 자동으로 GPU를 사용합니다.
대량의 이미지를 처리할 때는 배치 처리를 고려하세요.
모델의 성능은 학습 데이터와 가중치에 따라 달라질 수 있습니다.