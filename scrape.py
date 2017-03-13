#!/usr/bin/env python

# Dodgy script to capture images and tags.


import os.path
import re
import csv

import urllib2
import socks
import socket

import mechanize
import cookielib
from bs4 import BeautifulSoup

USERNAME = 'pubelle@gmail.com'
PASSWORD = 'poubelle'

OUTPUT_DIR = './data'
PICS_DIR = OUTPUT_DIR + '/images'
TAG_PATH = '/'.join([OUTPUT_DIR, 'tags.csv'])
BASE_URL = 'https://www.tag-walk.com/en/'
TAG_BASE = r'/en/photo/list/woman/all-categories/all-cities/all-seasons/all-designers/(.*)$'
TAG_REGEX = re.compile(TAG_BASE)

BASE_PHOTOS = 'https://www.tag-walk.com/en/photo/list/woman/all-categories/all-cities/all-seasons/all-designers/'

user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
headers = ('User-Agent', user_agent)

# docker pull negash/docker-haproxy-tor:latest
# docker run -d -p 5566:5566 -p 2090:2090 -e tors=25 negash/docker-haproxy-tor
# curl --socks5 192.168.99.100:5566 http://ifconfig.io
#socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, "192.168.99.100", 5566)
#socket.socket = socks.socksocket
#for i in range(1, 10):
    #print urllib2.urlopen('http://ifconfig.io').read()

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
    nb = (
        div
        .text
        .replace(' ', '')
        .replace('Results', '')
        .replace('Result', '')
    )
    return int(nb)

def main():
    cj = cookielib.LWPCookieJar()
    br = mechanize.Browser()
    br.set_cookiejar(cj)
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    br.set_handle_refresh(
        mechanize._http.HTTPRefreshProcessor(), max_time=1
    )
    br.addheaders = [headers]
    br.open("https://www.tag-walk.com/auth/en/login")

    br.select_form(nr=0)
    print br
    br.form['_username'] = USERNAME
    br.form['_password'] = PASSWORD
    br.submit()

    must_collect = False
    start_tag = 'bandana'

    for tag in tags:
        print tag
        if tag == start_tag:
            must_collect = True
        if must_collect:
            tag_path ='/'.join([PICS_DIR, tag])

            if not os.path.exists(tag_path):
                os.makedirs(tag_path)

            nb_results = get_tag_num_results(tag)
            print "%s results to collect" %(nb_results)
            img_counter = 0
            page_counter = 1

            while img_counter <= nb_results:
                url_format = BASE_PHOTOS + tag + '?page=%s' %(page_counter)
                print "Fetching %s" %(url_format)
                page = br.open(url_format).read()
                soup = BeautifulSoup(page, "lxml")
                soup.prettify()

                anchors = soup.findAll('div', {"class": "photoimg"})
                for anchor in anchors:
                    img_name = anchor.a.img['alt']
                    img_src = anchor.a.img['src']
                    path ='/'.join([tag_path, img_name])
                    print path
                    with open(path, 'w') as img_file:
                        #req = urllib2.Request(img_src, headers=headers)
                        img = br.open(img_src).read()
                        img_file.write(img)
                        img_counter = img_counter + 1

                page_counter = page_counter + 1

main()
