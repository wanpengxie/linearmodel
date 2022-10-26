wget https://s3-eu-west-1.amazonaws.com/reco-dataset/CriteoBannerFillingChallenge.tar.gz
tar -xvf train.tar.gz
mkdir -p data
python3 process_data.py dac/train.txt data

../trainer -v=3 -logtostderr -conf criteo.conf -parallel 10 -model ffm
