# Road_Sence_Detect
This project is detect the road line,car,pothole and trafficsign
# Train Code
python roadscene.py train --dataset=../roadscene_dataset/roadscene --weights=coco
# Predict
python roadscene.py show --dataset=../roadscene_dataset/roadscene --weights=mask_rcnn_roadscene_20.h5 --image=../roadscene_dataset/roadscene/val/img_23921.jpg
# Show
