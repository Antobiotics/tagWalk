import os.path
import re
import csv
import urllib2
from bs4 import BeautifulSoup

OUTPUT_DIR = './data'
PICS_DIR = OUTPUT_DIR + '/images'
TAG_PATH = '/'.join([OUTPUT_DIR, 'tags.csv'])
BASE_URL = 'https://www.tag-walk.com/en/'
TAG_BASE = r'/en/photo/list/woman/all-categories/all-cities/all-seasons/all-designers/(.*)$'
TAG_REGEX = re.compile(TAG_BASE)

BASE_PHOTOS = 'https://www.tag-walk.com/en/photo/list/woman/all-categories/all-cities/all-seasons/all-designers/'

user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
headers = { 'User-Agent' : user_agent }

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


def get_tag_num_results(tag):
    url_format = BASE_PHOTOS + tag + '?page=1'
    page = urllib2.urlopen(url_format).read()
    soup = BeautifulSoup(page, "lxml")
    soup.prettify()
    div = soup.find('div', {"class": "nbresult"})
    nb = div.text.replace(' ', '').replace('Results', '')
    return int(nb)

for tag in tags:
    print tag
    tag_path ='/'.join([PICS_DIR, tag])

    if not os.path.exists(tag_path):
        os.makedirs(tag_path)

    nb_results = get_tag_num_results(tag)

    img_counter = 0
    page_counter = 1

    while img_counter <= nb_results:
        url_format = BASE_PHOTOS + tag + '?page=%s' %(page_counter)
        print "Fetching %s" %(url_format)
        page = urllib2.urlopen(url_format).read()
        soup = BeautifulSoup(page, "lxml")
        soup.prettify()

        anchors = soup.findAll('div', {"class": "photoimg"})
        for anchor in anchors:
            img_name = anchor.a.img['alt']
            img_src = anchor.a.img['src']
            path ='/'.join([tag_path, img_name])
            print path
            with open(path, 'w') as img_file:
                req = urllib2.Request(img_src, headers=headers)
                img = urllib2.urlopen(req).read()
                img_file.write(img)
                img_counter = img_counter + 1

        page_counter = page_counter + 1
