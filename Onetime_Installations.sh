#!/bin/bash
#sudo -u ec2-user -i <<'EOF'

#source activate python3

echo "script start"
chmod 777 /tmp
apt-get update --allow-unauthenticated

pip install bcolz
pip install wand
pip install msgpack==0.5.6
apt-get install -y libsm6 libxext6 libxrender-dev
# apt-get -y install libx11-dev
# apt-get -y install zlib1g-dev
# apt-get -y install libpng12-dev
# apt-get -y install libjpeg-dev 
# apt-get -y install libfreetype6-dev
# apt-get -y install libxml2-dev
apt-get -y install libjpeg8
apt-get -y install libjpeg-turbo8
apt-get -y install libjpeg-turbo8-dev
apt-get -y install libjpeg62

pip install opencv-python
pip install graphviz
pip install sklearn
pip install sklearn-pandas
pip install isoweek
pip install pandas_summary
pip install keras
pip install tensorflow
pip install torchtext
pip install pytesseract
pip install fuzzywuzzy
pip install argparse

apt-get -y install libpng12-0 libpng12-dev
apt-get -y install libjpeg62 libjpeg62-dev
apt-get -y install libmagickwand-dev

apt-get -y install ghostscript

wget https://www.imagemagick.org/download/ImageMagick.tar.gz
#wget https://sourceforge.net/projects/imagemagick/files/old-sources/6.x/6.8/ImageMagick-6.8.9-10.tar.gz/download
tar -zxvf ImageMagick.tar.gz
cd ImageMagick*
./configure
make
make install
make distclean
sudo ldconfig

cd ..

# wget https://sourceforge.net/projects/libjpeg/files/libjpeg/6b/jpegsrc.v6b.tar.gz/download
# tar -zxvf jpegsrc.v6.tar.gz
# cd jpeg*
# make
# make install
# cd ..

#apt-get -y install imagemagick

echo "tesseract installtions start"

apt-get -y install tesseract-ocr

cd /usr/local/share/tessdata
wget https://github.com/tesseract-ocr/tessdata/raw/master/eng.traineddata
wget https://github.com/tesseract-ocr/tessdata/blob/master/osd.traineddata
wget https://github.com/tesseract-ocr/tessdata/blob/master/equ.traineddata

export TESSDATA_PREFIX=/usr/local/share
export PYTHONIOENCODING=utf8
#tesseract --list-langs

echo "tesseract installtions completed"

#source deactivate
#EOF
