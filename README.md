# Road_Sence_Detect
This project is detect the road line,car,pothole and trafficsign
# Datasets
This datasets labeled by VGG Image Annotator [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/)
# Train Code
python roadscene.py train --dataset=../roadscene_dataset/roadscene --weights=coco
# Predict
Image:<br>
python roadscene.py show --dataset=../roadscene_dataset/roadscene --weights=mask_rcnn_roadscene_20.h5 --image=../roadscene_dataset/roadscene/val/img_23921.jpg <br>
Video:<br>
python process_video.py 
# Show
[Image](https://github.com/VincentMy/Road_Sence_Detect/blob/master/output/detected_20200419T092125.png.jpg)
[Image](https://github.com/VincentMy/Road_Sence_Detect/blob/master/output/detected_20200419T091814.png.jpg)
