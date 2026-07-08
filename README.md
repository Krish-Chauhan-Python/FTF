# FTF - Vision-Based Torque Estimation

This repository implements all 7 model architectures from the research paper on vision-based torque estimation for robotic grasping using the RH20T dataset.

## Installation

`ash
pip install -r requirements.txt
`

## Dataset Preparation
Set your dataset path in .env:
`ash
cp .env.example .env
# Edit .env with your path to RH20T_cfg1/Chemistry
`

## Quick Start
`ash
python scripts/train.py --model yolo_pipeline --epochs 25
python scripts/evaluate.py --model yolo_pipeline --weights weights/best_yolo_regressor.pth
`
