from pymongo import MongoClient
# import pymongo
# import pprint as pp
from bs4 import BeautifulSoup
import requests
from time import sleep
import sys
import re
from datetime import date
from selenium import webdriver
# import webkit_server
# import dryscrape
from math import ceil
from collections import defaultdict
import threading
import subprocess


def get_last_page(url):
    try:
        r1 = requests.get(url)
    except:
        return "Failed to get request from url:", sys.exc_info()[0]
    soup_last_page = BeautifulSoup(r1.text).find(title='last page')
    if soup_last_page:
        last_page = int(re.findall('\w+',soup_last_page.text)[0])
        return last_page
    else:
        raise Exception, "last_page title attribute not found in html"


def add_game_db(mongodb_collection_games,url=None, post_url=None, root_url=None, page_end=None, testing=True):

    if url==None:
        url = 'https://boardgamegeek.com/search/boardgame/page/'
    if post_url==None:
        post_url = '?sort=numvoters&advsearch=1&q=&include%5B' + \
            'designerid%5D=&include%5Bpublisherid%5D=&' + \
            'geekitemname=&range%5Byearpublished%5D%5B' + \
            'min%5D=&range%5Byearpublished%5D%5Bmax%5D=&range%5B' + \
            'minage%5D%5Bmax%5D=&range%5Bnumvoters%5D%5Bmin%5D=&range%5B' + \
            'numweights%5D%5Bmin%5D=&range%5Bminplayers%5D%5B' + \
            'max%5D=&range%5Bmaxplayers%5D%5Bmin%5D=&range%5B' + \
            'leastplaytime%5D%5Bmin%5D=&range%5Bplaytime%5D%5B' + \
            'max%5D=&floatrange%5Bavgrating%5D%5Bmin%5D=&floatrange%5B' + \
            'avgrating%5D%5Bmax%5D=&floatrange%5Bavgweight%5D%5B' + \
            'min%5D=&floatrange%5Bavgweight%5D%5Bmax%5D=&' + \
            'colfiltertype=&searchuser=&nosubtypes%5B0%5D=boardgameexpansion&playerrangetype=normal&' + \
            'B1=Submit&sortdir=desc'  #advanced filter with all options set to blank except
                                      #boardgame-expansions and output in descending order by num_voters
    if root_url==None:
        root_url = 'https://boardgamegeek.com'

    if page_end==None:
        page_end = get_last_page(url+'1'+post_url) #the url requires a page number. The choice of page 1 is arbitrary

    i=0  #only using for testing


    for page_num in xrange(1,page_end+1):
        if testing==True:
            i += 1
            if i > 10:
                return "testing=True ended the program at i greater than 2"
        url_ = url + str(page_num) + post_url
        sleep(0.025)
        r = requests.get(url_)
        print "working on page {} out of {}".format(page_num,page_end)
        soup = BeautifulSoup(r.text)
        table = soup.find('table','collection_table')
        rows = table.find_all(id='row_')
        for row in rows:
            game_url_name_tag = row.find('td','collection_objectname').find('a')
            game_name = game_url_name_tag.text
            game_url = root_url + game_url_name_tag.get('href')
            num_voters = row.find_all('td','collection_bggrating')[-1].text
            try:
                num_voters = int(num_voters)
            except ValueError:
                num_voters = 0
            except:
                num_voters = 0.0001  #leaving a numerical value to be safe but docstring shall have a comment about this
            game_id = int(re.findall(r'([0-9]+)/.*$',game_url)[0]) #extract game_id from url
            mongodb_collection_games.insert_one({'name':game_name,
                                                 'num_voters':num_voters,
                                                 'game_url':game_url,
                                                 'game_id':game_id})


def page_loading(sess_body_obj):
    soup1 = BeautifulSoup(sess_body_obj,'html')
    try:
        soup1.find_all('li','summary-item summary-rating-item')[0]
        return True
    except:
        return False


def is_number(string):
    '''
    Returns 0 if text cannot be converted to float
    '''
    try:
        return float(string)
    except:
        return 0






def visit_game_pg(job_list, machine_num, mongodb_collection_all_info, search_date, testing=True,start_page=None):

    try:
        i=0
        j=0
        for game in job_list[machine_num]:
            if not game.get('search_date') or game.get('search_date') < search_date:
                page=1 #just here for except clause handling
                index = 0 #just here for except clause handling
                print "not searched yet, searching now...."
                url = game['game_url'] + '/ratings?rated=1&pageid=1'
                #rated=1 filters out users that just commented but didn't rate
                print "game_url is {}".format(url)
                #https://boardgamegeek.com/boardgame/13/catan/ratings?rated=1&pageid=1
                #
                # capabilities = webdriver.DesiredCapabilities().FIREFOX
                # capabilities["marionette"] = False
                # browser = webdriver.Firefox(capabilities=capabilities)
                driver = webdriver.Firefox()
                driver.get(url)
                source = driver.page_source

                soup = BeautifulSoup(source,'html')
                # sess.reset()
                last_page = soup.find_all('div','outline-item-description')[1].find('a').text
                last_page = int(re.sub("\W","",last_page))
                last_page = int(ceil(last_page*1./100))    #100 listings per page so 69145/100 --> 692, this page will have the last of it
                if start_page==None:
                    start_page = 1
                for page in xrange(start_page,last_page+1):
                    url_page = game['game_url'] + ('/ratings?rated=1&pageid={}'.format(page))

                    # driver = webdriver.Firefox()
                    driver.get(url_page)
                    source = driver.page_source
                    print "extracting users & ratings from ratings page {} of total pages {}".format(page,last_page)

                    index += 1 #just here for except clause handling

                    soup1 = BeautifulSoup(source,'html')
                    summary_item = soup1.find_all('li','summary-item summary-rating-item')

                    print url_page

                    for rating in summary_item:
                        player_rating = rating.find('div','summary-item-callout').text
                        player_name = rating.find('div','comment-header-title').find('a').text
                        mongodb_collection_all_info.insert_one({'game_name':game['name'],
                                                                'game_id':game['game_id'],
                                                                'player_name':player_name,
                                                                'player_rating':is_number(player_rating)})
                    start_page=None #so the next game can start at page 1

                    if testing==True:
                        i += 1
                        if i > 5:
                            j += 1
                            if j>5:
                                return "stopping after trying five games (five pages each) Testing=True,j=5"
                            break
                            return "Stopped after 5 pages because Testing = True, func visit_game_pg"
                game['search_date'] = date.today()
    except:
        # subprocess.call('/home/ubuntu/bash.sh')
        job_list[machine_num] = job_list[machine_num][index:]
        # visit_game_pg(job_list,machine_num,all_info,date(2017,7,1),testing=False,start_page=page)
        raise ValueError

def mid_game_visit_game_pg(game, mongodb_collection_all_info,start_page=1):
            url = game['game_url'] + '/ratings?rated=1&pageid=1'
            driver = webdriver.Firefox() #firefox, needs to be installed
            driver.get(url)
            source = driver.page_source

            soup = BeautifulSoup(source,'html')
            last_page = soup.find_all('div','outline-item-description')[1].find('a').text
            last_page = int(re.sub("\W","",last_page))
            last_page = int(ceil(last_page*1./100))
            for page in xrange(start_page,last_page+1):
                url_page = game['game_url'] + ('/ratings?rated=1&pageid={}'.format(page))

                # driver = webdriver.Firefox() #firefox, needs to be installed
                driver.get(url_page)
                source = driver.page_source
                print "extracting users & ratings from ratings page {} of total pages {}".format(page,last_page)

                soup1 = BeautifulSoup(source,'html')
                summary_item = soup1.find_all('li','summary-item summary-rating-item')

                print url_page

                for rating in summary_item:
                    player_rating = rating.find('div','summary-item-callout').text
                    player_name = rating.find('div','comment-header-title').find('a').text
                    mongodb_collection_all_info.insert_one({'game_name':game['name'],
                                                            'game_id':game['game_id'],
                                                            'player_name':player_name,
                                                            'player_rating':is_number(player_rating)})

def mach_jobs(num_machines,mongodb_collection_games):
    games = mongodb_collection_games
    job_list = defaultdict(list)
    bucket = 0
    machine_num = 1
    machine_cap = sum([int(ceil(game['num_voters']*1./100)) for game in games.find()])/num_machines #100 reviews per page
    for game in games.find():
        job_list[machine_num] += [game]
        num_pgs = int(ceil(game['num_voters']*1./100))  #num_pgs of reviews
        bucket += num_pgs
        if bucket > (machine_cap)*0.95:
                machine_num += 1
                bucket = 0
    return job_list

def start_midgame(job_list,machine_num,all_info,game_id,start_page):
    index=0
    for game in job_list[machine_num]:
        if game['game_id']==game_id:
            break
        index+=1
    job_list[machine_num] = job_list[machine_num][index:]
    game = job_list[machine_num][0]
    visit_game_pg(job_list, machine_num, all_info, date(2017,7,1), testing=False,start_page=start_page)


if __name__ == '__main__':
    client = MongoClient()
    db = client['bgg_data']
    games = db['games']
    all_info = db['all_info']

    # add_game_db(games,testing=False)
    num_machines = 30
    # machine_num =
    # job_list = mach_jobs(num_machines,games)
    # visit_game_pg(job_list,machine_num,all_info,date(2017,7,1),testing=False)
    # mid_game_visit_game_pg(game,all_info,start_page=126)




    # machine_num = 1 #running on ec2-18-220-7-156.us-east-2.compute.amazonaws.com  --started on 1:37am, 07/11/2017    ip-172-31-30-247
        #Done collecting

    # machine_num = 2 #running on  ec2-13-59-255-119.us-east-2.compute.amazonaws.com --started on 2:03am, 07/11/2017    ip-172-31-19-213
        #Done collecting

    # machine_num = 3 #running on  ec2-13-59-85-194.us-east-2.compute.amazonaws.com --started on 1:55am, 07/11/2017    ip-172-31-29-121
        #Done collecting, all_info.count() = 162,200 --> on S3

    # machine_num = 4 #running on  ec2-13-58-105-212.us-east-2.compute.amazonaws.com --started on 10:45am, 07/11/2017    ip-172-31-28-91
        #Done collecting, all_info.count() = 155,100 --> on S3

    # machine_num = 5 #running on  ec2-13-59-137-129.us-east-2.compute.amazonaws.com --started on 10:47am, 07/11/2017    ip-172-31-24-233
        #Done collecting, all_info.count() = 149,250 --> on S3

    # machine_num = 6 #running on  ec2-52-15-126-217.us-east-2.compute.amazonaws.com --started on 11:18am, 07/11/2017    ip-172-31-3-92
        #Done collecting, all_info.count() = 154,650 --> on S3

    # machine_num = 7 #running on  ec2-13-58-211-20.us-east-2.compute.amazonaws.com --started on 11:21am, 07/11/2017    ip-172-31-4-130
        #Done collecting

    # machine_num = 8 #running on  ec2-13-58-62-79.us-east-2.compute.amazonaws.com --started on 12:30pm, 07/11/2017    ip-172-31-3-239
        #Done collecting

    # machine_num = 9 #running on  ec2-18-220-54-59.us-east-2.compute.amazonaws.com --started on 2:32pm, 07/11/2017    ip-172-31-12-40
        #Done collecting

    # machine_num = 32 #running on  ec2-13-59-74-161.us-east-2.compute.amazonaws.com --started on 1:32pm, 07/11/2017    ip-172-31-14-170   #expected to finish in 1.5 hrs max
        #Done collecting, all_info.count() = 20,150 --> on S3

    # machine_num = 15 #running on  ec2-13-59-74-161.us-east-2.compute.amazonaws.com --started on 7:22pm, 07/11/2017    ip-172-31-14-170
        #Done collecting

    # machine_num = 16 #running on  ec2-13-59-85-194.us-east-2.compute.amazonaws.com --started on 7:25pm, 07/11/2017    ip-172-31-29-121
        #Done collecting

    # machine_num = 17 #running on  ec2-13-58-105-212.us-east-2.compute.amazonaws.com --started on 7:27pm, 07/11/2017    ip-172-31-28-91
        #Done collecting

    # machine_num = 18 #running on  ec2-13-59-137-129.us-east-2.compute.amazonaws.com --started on 7:27pm, 07/11/2017    ip-172-31-24-233
        #Done collecting

    # machine_num = 19 #running on  ec2-52-15-126-217.us-east-2.compute.amazonaws.com --started on 9:40pm, 07/11/2017    ip-172-31-3-92
        #Done collecting

    # machine_num = 20 #running on  ec2-13-59-255-119.us-east-2.compute.amazonaws.com --started on 2:30pm, 07/12/2017    ip-172-31-19-213
        #Done collecting

    # machine_num = 21 #running on  ec2-13-58-105-212.us-east-2.compute.amazonaws.com --started on 2:30pm, 07/12/2017    ip-172-31-28-91
        #Done collecting

    # machine_num = 22_1st_half #running on  ec2-13-58-62-79.us-east-2.compute.amazonaws.com --started on 2:30pm, 07/12/2017    ip-172-31-3-239
        #Only collected 1st_half upto game_id 35503, started the rest on ip 172-31-19-213

    # machine_num = 22_2nd_half #running on  ec2-13-59-255-119.us-east-2.compute.amazonaws.com --started on 1:30am, 07/13/2017    ip-172-31-19-213
        #Done collecting

    # machine_num = 23 #running on  ec2-13-59-137-129.us-east-2.compute.amazonaws.com --started on 2:37am, 07/12/2017    ip-172-31-24-233
        #Done collecting

    # machine_num = 24 #running on  ec2-18-220-54-59.us-east-2.compute.amazonaws.com --started on 2:40pm, 07/12/2017    ip-172-31-12-40
        #Done collecting

    # machine_num = 25 #running on ec2-18-220-7-156.us-east-2.compute.amazonaws.com  --started on 2:40pm, 07/12/2017    ip-172-31-30-247
        #Done collecting

    # machine_num = 26 #running on  ec2-13-59-85-194.us-east-2.compute.amazonaws.com --started on 3:45pm, 07/12/2017    ip-172-31-29-121
        #Done collecting

    # machine_num = 27 #running on  ec2-13-59-137-129.us-east-2.compute.amazonaws.com --started on 12:30am, 07/13/2017    ip-172-31-24-233
        #Done collecting

    # machine_num = 28 #running on  ec2-13-59-137-129.us-east-2.compute.amazonaws.com --started on 2:30pm, 07/13/2017    ip-172-31-24-233
# asd
    # machine_num = 29 #running on  ec2-13-58-105-212.us-east-2.compute.amazonaws.com --started on 12:30am, 07/13/2017    ip-172-31-28-91
        #Done collecting

    # machine_num = 30 #running on  ec2-13-59-85-194.us-east-2.compute.amazonaws.com --started on 12:30am, 07/13/2017    ip-172-31-29-121
        #Done collecting

    # machine_num = 31 #running on  ec2-52-15-126-217.us-east-2.compute.amazonaws.com --started on 10:00am, 07/13/2017    ip-172-31-3-92
# ads
    # machine_num = 10 #running on ec2-18-220-7-156.us-east-2.compute.amazonaws.com  --started on 10:05am, 07/13/2017    ip-172-31-30-247
        #Done collecting

    # machine_num = 11 #running on  ec2-13-58-105-212.us-east-2.compute.amazonaws.com --started on 12:30am, 07/13/2017    ip-172-31-28-91
        #Done collecting

    # machine_num = 12 #running on  ec2-18-220-54-59.us-east-2.compute.amazonaws.com --started on 12:40pm, 07/13/2017    ip-172-31-12-40
        #Done collecting

    # machine_num = 13 #running on  ec2-13-59-255-119.us-east-2.compute.amazonaws.com --started on 1:00pm, 07/13/2017    ip-172-31-19-213
        #Done collecting

    # machine_num = 14 #running on  ec2-13-59-74-161.us-east-2.compute.amazonaws.com --started on 1:22pm, 07/13/2017    ip-172-31-14-170
        #Done collecting

    # Analysis can also run on 172-31-3-92 machine with 32 G ram  ec2-52-15-126-217.us-east-2.compute.amazonaws.com
    # Analysis machine #running on  ec2-13-59-85-194.us-east-2.compute.amazonaws.com ip-172-31-29-121
    # Analysis now running on ec2-18-220-7-156.us-east-2.compute.amazonaws.com  --started on 9:20pm, 07/13/2017    ip-172-31-30-247!pip
