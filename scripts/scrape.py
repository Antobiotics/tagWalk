#!/usr/bin/env python
import os.path
import re
import csv
import json

import random
from time import sleep

import urllib2
import httplib
import socket

import mechanize
import cookielib

from bs4 import BeautifulSoup

USERNAME = 'pubelle@gmail.com'
PASSWORD = 'poubelle'

OUTPUT_DIR = '../tag_walk/data/tag_walk/'
MEMORY_PATH = OUTPUT_DIR + 'crawl_memory.json'
PICS_DIR = OUTPUT_DIR + '/images/v2'


TAG_PATH = '/'.join([OUTPUT_DIR, 'tags.csv'])


BASE_URL = 'https://www.tag-walk.com/en/'
TAG_BASE = r'/en/photo/list/woman/all-categories/all-cities/all-seasons/all-designers/(.*)$'
TAG_REGEX = re.compile(TAG_BASE)

BASE_PHOTOS = 'https://www.tag-walk.com/en/photo/list/woman/all-categories/all-cities/all-seasons/all-designers/'

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
headers = ('User-Agent', user_agent)


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

def set_browser():
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
    br.form['_username'] = USERNAME
    br.form['_password'] = PASSWORD
    br.submit()

    return br


class TagWalkCrawler():
    def __init__(self, reset_mem=False):
        self.browser = set_browser()
        self.reset_mem = reset_mem

        self.tags = get_tags()

        self.memory_path = MEMORY_PATH
        self.set_memory()

    def save_memory(self):
        with open(self.memory_path, 'w') as mem_file:
            mem_file.write(json.dumps(self.memory))

    def set_memory(self):
        try:
            with open(self.memory_path, 'r') as mem_file:
                self.memory = json.load(mem_file)
                if self.reset_mem:
                    for key in self.memory:
                        self.memory[key]['done'] = False
        except Exception as e:
            print e
            self.memory = {}

    def get_unprocessed_tags(self):
        unprocessed_tags = []
        for tag in self.tags:
            done = False
            try:
                done = self.memory[tag]['done']
            except KeyError:
                pass

            if not done:
                unprocessed_tags.append(tag)
        return unprocessed_tags

    def mk_tag_dir(self, tag):
        tag_path ='/'.join([PICS_DIR, tag])
        if not os.path.exists(tag_path):
            os.makedirs(tag_path)
        return tag_path

    def update_memory(self, tag_desc):
        self.memory[tag_desc['name']] = tag_desc
        self.save_memory()


    def fetch_data(self, tag_desc):
        img_counter = 0

        nb_results = tag_desc['num_images']
        skip = False
        if nb_results == 36277 or tag_desc['name'] == 'black-trousers':
            skip = True

        if not skip:
            while img_counter <= nb_results:
                url_format = BASE_PHOTOS + tag_desc['name'] + '?page=%s' %(tag_desc['current_page'])
                print "Fetching %s" %(url_format)
                page = self.browser.open(url_format).read()
                soup = BeautifulSoup(page, "lxml")
                soup.prettify()

                anchors = soup.findAll('div', {"class": "photoimg"})
                for anchor in anchors:
                    href = anchor.a['href']

                    image_desc = {
                        'name': anchor.a.img['alt'],
                        'href': anchor.a['href'],
                        'season': href.split('/')[5],
                        'designer': href.split('/')[6],
                        'src': anchor.a.img['src'],
                        'path': '/'.join([tag_desc['local_path'],
                                          anchor.a.img['alt']])
                    }

                    processed_src = [image['src'] for image in tag_desc['images']]
                    print "Already processed: %s/%s" %(len(processed_src), nb_results)
                    if not image_desc['src'] in processed_src:
                        with open(image_desc['path'], 'w') as img_file:
                            try:
                                img = self.browser.open(image_desc['src']).read()
                                img_file.write(img)
                                img_counter = img_counter + 1

                                tag_desc['images'].append(image_desc)
                                self.update_memory(tag_desc)

                                sleep_time = random.uniform(0, 1)
                                print "Sleeping %d" %(sleep_time)
                                sleep(sleep_time)

                            except httplib.BadStatusLine as e:
                                print "BAD Status Error: %s" % e
                                self.update_memory(tag_desc)

                            except Exception as e:
                                print "UNKOWN Error: %s" % e
                                self.update_memory(tag_desc)
                                return tag_desc
                    else:
                        img_counter = img_counter + 1

                tag_desc['current_page'] = tag_desc['current_page'] + 1

        tag_desc['done'] = True
        self.update_memory(tag_desc)
        return tag_desc


    def run(self):
        to_process_tags = self.get_unprocessed_tags()
        for tag_name in to_process_tags:
            tag_path = self.mk_tag_dir(tag_name)
            nb_results = get_tag_num_results(tag_name)
            print "%s results to collect" %(nb_results)

            tag_descriptor = {
                'current_page': 1,
                'done': False,
                'skipped': False,
                'name': tag_name,
                'num_images': nb_results,
                'local_path': tag_path,
                'images': []
            }
            if tag_name in self.memory.keys():
                tag_descriptor = self.memory[tag_name]

            tag_desc = self.fetch_data(tag_descriptor)
            print tag_desc



if __name__ == "__main__":
    crawler = TagWalkCrawler(reset_mem=True)
    try:
        crawler.run()
    except KeyboardInterrupt:
        print "Abort!!!! Save Memory First!"
        #print crawler.memory
        crawler.save_memory()
