######################### DOWNLOADING REQUIRED FILES AND DATASET #########################
# Create dataset folder
cd ..
mkdir dataset
mkdir dataset/COCO/
cd dataset/COCO/

# Clone COCO API
git clone https://github.com/pdollar/coco.git

# Create required folders in dataset/COCO/
mkdir images
mkdir images/mask2014
mkdir json
mkdir mat

# Download oficial COCO dataset
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip

# Unzip oficial COCO dataset
unzip person_keypoints_trainval2014.zip -d .
unzip val2014.zip -d ./images
unzip test2014.zip -d ./images
unzip train2014.zip -d ./images
unzip test2015.zip -d ./images

# Optional - Save space by removing original zip files
# rm -f person_keypoints_trainval2014.zip
# rm -f test2015.zip
# rm -f test2014.zip
# rm -f train2015.zip
# rm -f val2014.zip
