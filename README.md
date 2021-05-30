# Projekt KNN
### Téma: Odhad absolútnej hĺbky z obrazu (na prírodných dátach)

### Autoři: Adam Ferencz, Tomáš Mojžiš, Petr Mičulek

# Useful links
* [Milanote tabule](https://app.milanote.com/1Le9SR1KDLsYe6/projekt-knn?p=euz2wFNepjw)
* [cphoto depth paper](http://cphoto.fit.vutbr.cz/depth/)
* [GeoPose3k](http://cphoto.fit.vutbr.cz/geoPose3K/)
* [megadepth paper](https://www.cs.cornell.edu/projects/megadepth/)

# Project structure
Some parts of this project are taken from other repositories:
* `semseg/` - this directory is taken from <https://github.com/CSAILVision/semantic-segmentation-pytorch>
               and it's not used in our implementation, we only used it at the beginning
* `geopose/model/houglass_ugly.py` - contains the original model from <https://github.com/zhengqili/MegaDepth>,
                                      we rewrote it because it was pretty ugly
* `geopose/model/hourglass.py` - contains the same model, but implemented in pytorch from scratch,
                                 inspired by <https://github.com/dfan/single-image-surface-normal-estimation>
* `geopose/model/hourglass_*.py` - contains derived models from base HG with added FOV
* `geopose/train.py` - Working with DistributedDataParallel <https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c>  

Other files are scripts implemented by us. Main scripts are:
* `geopose/train.py` - training script, containing also evaluating on test dataset
* `geopose/dataset.py` - script containing DataLoader class
* `geopose/losses.py` - script containing implementation of loss functions
* and other minor utility scripts

### Notes

Run from root repo directory (KNN-Projekt).

Loading JPEG uses TurboJPEG library that must be installed through a package manager.

`sudo apt install libturbojpeg`

Python package dependencies can be installed from requirements.txt.

`pip install -r requirements.txt`

### How to launch
* install requirements.txt with pip
* for training on `datasets/geoPose3K_final_publish` run `python geopose/train.py`
    * set the parameters inside the script
* for plotting run `python plot_predictions.py` - set the `weights_path` variable
  * it could not be passed as parameter because of megadepth model builder uses argparse and it interferences
