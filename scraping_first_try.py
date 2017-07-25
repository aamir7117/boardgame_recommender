from pymongo import MongoClient
import pymongo
import pprint as pp
from bs4 import BeautifulSoup
import requests
import pprint
from time import sleep
import sys
import re
import psycopg2
from datetime import date


def searched_users_to_db(mongodb_collection_users, mongodb_collection_pages, url='https://boardgamegeek.com/users/page/', page_start=1, page_end=None, testing=False):
    '''
    The Boardgamegeek website displays only 25 users per page as of July, 2017. This function will step 1 unit from
    page_start to page_end and insert the returned webpage text into the supplied db.
    page_end will be appended to the end of the url string iteratively
    if page_end is set to None, last_page will be extracted automatically from the url, if possible
    '''

    page_end = get_last_page(url + str(page_start))  # the url only works if a page num is supplied

    for page_num in xrange(page_start, page_end+1):
        if testing==True:
            if page_num > 25:
                print "tested until page 5"
                break
        sleep(0.025)
        url_ = url + str(page_num)
        r = requests.get(url_)
        print "inserting searched user page number {}".format(page_num)
        # mongodb_collection.insert_one({'html':r.text, 'url':url_})
        mongodb_collection_pages.insert_one({'url':url_})
        extract_users_to_db(mongodb_collection_users,html_text=r.text)  #call each user's page right then...




def extract_users_to_db(mongodb_collection_users, mongodb_collection_pages=None, \
                            html_text='<html>....</html>', \
                            root_users_url='https://boardgamegeek.com/collection', \
                            post_path='?rated=1&subtype=boardgame'):
    '''
    for each page in mongo_collection_pages, find all users on that page and insert
    that users' info into mongodb_collection_users.
    Among the attributes saved for each user are: username, country (text) and
    ratings_url (which is concatenated like so: root_users_url + extracted_from_page_url + post_path)
    '''

    if mongodb_collection_pages == None:
        if html_text=='<html>....</html>':
            raise Exception, "if no mongodb_collection_pages is supplied, supply html_text please"
        else: iterable = xrange(1)
    else:
        iterable = pymongo.cursor.Cursor(mongodb_collection_pages)


    for page in iterable:
        if mongodb_collection_pages==None:
            text = html_text
        else:
            text = page['html']
        text = re.sub('[(\r)(\n)(\t)]',"",text)
        soup = BeautifulSoup(text)
        users_in_page = soup.find_all('div','avatarblock js-avatar ')

        for user_field in users_in_page:

            user = user_field.find_next('div','username')

            username = user.text
            ratings_url = root_users_url + user.find_next('a').get('href') + post_path
            country = user_field.find("div","location js-location").text

            mongodb_collection_users.insert_one({'username' : username, \
                                                 'ratings_url' : ratings_url, \
                                                 'country' : country })

def extract_ratings_to_db(mongodb_collection_users,testing=True):
    i = 0
    for user in pymongo.cursor.Cursor(mongodb_collection_users):
        if not user.get('ratings'):
            r = requests.get(user.get('ratings_url'))
            user['ratings'] = r.text
            i += 1
            if i > 25:
                print "i is greater than 5, testing=True"
                break

if __name__ == '__main__':
    client = MongoClient()
    db = client['bgg_data']
    searched_pages = db['users_searched']
    users_db = db['all_bgg_users']
    # games = db['games']
    searched_users_to_db(searched_pages, testing=True)
    # extract_users_to_db(searched_pages, users_db)
