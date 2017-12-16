TAG_WALK
========


__TODO:__ https://github.com/Antobiotics/tagWalk/projects/1

Description:
============

TagWalk is a set of experiments around fasion data. The goal is to build style annotation, clothes segmentation and style similarity search algorithms and to expose them in a react-native application.

Installation:
=============

stuff around `fachung.cfg` here. 

Datasets:
=========

1 - Tagwalk: Scraped data from tagwalk.com
2 - Asos: Scraped data from Asos.com
3 - Fashionista: data from the fashionista segmentation dataset.

Data is stored on s3 and can be pulled using (access keys available upon request, or run the scrapers yourself ;) ):

```
> python3 fachung.py managers s3_data --sync --pull
```

Training:
=========

