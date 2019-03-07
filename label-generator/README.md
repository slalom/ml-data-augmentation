



## Download and extract images
Preferably from: SUN Database https://groups.csail.mit.edu/vision/SUN/hierarchy.html

The 2012 database has 16,873 images.

Download SUN Database image archive:
```
wget http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
```

Extract archive into folder:
```
python3 extractbkgs.py --archive SUN2012.tar.gz --folder-name backgrounds
```

## Label generation
Once backgrounds are extracted into a folder, the following commands can be used to generate labels

```
python3 gen.py --make-num <number of labels>
```

Specify different logo:
```
python3 gen.py --make-num 3 --logo-path ./images/slalom_horizontal.jpg 
```

Also generate anomalies:
```
python3 gen.py --make-num 3 --logo-path ./images/slalom_horizontal.jpg --class-names slalom,anomalous_label
```

Custom Backgrounds folder:
```
python3 gen.py --make-num 3 --logo-path ./images/slalom_horizontal.jpg --class-names slalom,anomalous_label --backgrounds-dir custombkg/
```

Turn on debug to highlight bounding boxes:
```
python3 gen.py --make-num 3 --logo-path ./images/slalom_horizontal.jpg --class-names slalom,anomalous_label --debug
```
