# Projekt KNN
### Téma: Odhad absolútnej hĺbky z obrazu (na prírodných dátach)

### Autoři: Adam Ferencz, Tomáš Mojžiš, Petr Mičulek

# Useful links
* [Milanote tabule](https://app.milanote.com/1Le9SR1KDLsYe6/projekt-knn?p=euz2wFNepjw)
* [cphoto depth paper](http://cphoto.fit.vutbr.cz/depth/)
* [GeoPose3k](http://cphoto.fit.vutbr.cz/geoPose3K/)
* [megadepth paper](https://www.cs.cornell.edu/projects/megadepth/)

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
