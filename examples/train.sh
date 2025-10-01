#!/bin/bash
yolo detect train data=wideband_yolo.yaml model=yolo11s pretrained=yolo11s.pt device=0 epochs=1 batch=32 save=True save_period=1 single_cls=True imgsz=1024 name=11s_freeze1 cos_lr=False cache=False workers=32 freeze=1 lr0=0.0001 optimizer=SGD
