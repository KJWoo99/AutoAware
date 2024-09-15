import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from models.retinaface import RetinaFace
from data import cfg_mnet, cfg_re50

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm

# 설정값 정의
cfg = cfg_mnet  # MobileNet 설정 사용 (cfg_mnet) 또는 ResNet50 설정 사용 가능 (cfg_re50)
resize = 1
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750
vis_thres = 0.6

def detection(img, weights_path):
    # GPU 사용 가능 시 GPU 사용, 아니면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # RetinaFace 모델 로드 및 평가 모드로 설정
    model = RetinaFace(cfg, phase='test').to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 이미지 전처리
    img = np.float32(img)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)  # 평균 값 빼기
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # 모델 추론
    loc, conf, landms = model(img)

    # 결과 후처리
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # 낮은 신뢰도 결과 제거
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # NMS 전 상위 K개 결과 유지
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # NMS (Non-Maximum Suppression) 적용
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # 상위 K개 결과만 유지
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    # 최종 결과 생성 (경계 상자 + 랜드마크)
    dets = np.concatenate((dets, landms), axis=1)
    
    return dets