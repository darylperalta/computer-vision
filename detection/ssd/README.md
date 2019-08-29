## SSD Model Training

The train and test datasets including labels in csv format can be downloaded from this link:

[Dataset](https://bit.ly/30ldKcP)

In the ssd folder, create the dataset folder, copy the downloaded file there, and extract by running:

`mkdir dataset`

`cp drinks.tar.gz dataset`

`cd dataset`

`tar zxvf drinks.tar.gz`

The SSD model is trained for 200 epochs by executing:

`python3 ssd.py -t -b=4 -l=4 -n`

## SSD Model Validation

`python3 ssd.py --weights=<weights_file> -n  -l=4 -e  --image_file=<target_image_file>`

