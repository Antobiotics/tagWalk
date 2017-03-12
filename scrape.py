import os.path
import re
import csv
import urllib2
from bs4 import BeautifulSoup

OUTPUT_DIR = './data'
TAG_PATH = '/'.join([OUTPUT_DIR, 'tags.csv'])
BASE_URL = 'https://www.tag-walk.com/en/'
TAG_BASE = r'/en/photo/list/woman/all-categories/all-cities/all-seasons/all-designers/(.*)$'
TAG_REGEX = re.compile(TAG_BASE)



def find_tags():
    print "Searching for tags in %s" %(BASE_URL)
    page = urllib2.urlopen(BASE_URL).read()
    soup = BeautifulSoup(page, "lxml")
    soup.prettify()
    for anchor in soup.findAll('a', href=True):
        href = anchor['href']
        tag = TAG_REGEX.match(href)
        if not tag is None:
            yield tag.group(1)

def save_tags(tags, path=TAG_PATH):
    print "Saving tags to: %s" %(path)
    with open(path, 'w') as outfile:
        for t in tags:
            outfile.write(t + '\n')

def read_tags(path=TAG_PATH):
    print "Reading tags from: %s" %(path)
    tags = []
    with open(path, 'r') as tagfile:
        reader = csv.reader(tagfile, delimiter=',')
        for row in reader:
            tags.append(row[0])
    return tags

def get_tags(path=TAG_PATH):
    if not os.path.isfile(path):
        tags = find_tags()
        save_tags(tags, path=path)
        return list(tags)
    return read_tags(path=path)

tags = get_tags()
print tags
