import datetime
import os
import queue
import requests
import time
import threading
import twint

### Some problems with running on ipynb
# import nest_asyncio
# nest_asyncio.apply()

from argparse import ArgumentParser
from PIL import Image

def make_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--max_imgs', dest='max_img', default=20, type=int,
                        help="Number of images to download")
    parser.add_argument('-w', '--num_workers', dest='no_workers', default=6, type=int,
                        help="Number of worker threads to run with")
    parser.add_argument('-t', '--hashtag', dest='hashtag', default="nature", type=str,
                        help="Name of hashtag to sample images from")
    parser.add_argument('-r', '--resolution', dest='max_reso', default=1024, type=int,
                        help="Maximum resolution of image")
    args = parser.parse_args()
    return args

args = make_args()
HASHTAG = args.hashtag
NUM_TWEETS = args.max_img
MAX_WORKERS = args.no_workers
MAX_RESOLUTION = (args.max_reso, args.max_reso)
MAX_WORKERS = 6
job_pool = queue.Queue()
req_header = { "User-Agent": "CS4243 crawler bot", "From": "nnk@u.nus.edu" }

now = datetime.datetime.now()
datafile = open(f"tweet_crawl_{now.month:02}{now.day:02}_{HASHTAG}_test.csv", 'w')
os.makedirs(f"./hashtag/{HASHTAG.lower()}", exist_ok=True)

def get_image(url, filename) :
    req = requests.get(url, stream=True)
    if not req.ok:
        return
    with open(filename, 'wb') as f:
        for chunk in req.iter_content(1024):
            if chunk:
                f.write(chunk)
    with Image.open(filename) as im:
        im.thumbnail(MAX_RESOLUTION)
        im = im.convert("RGB")
        im.save(filename[:-3]+"jpeg", "JPEG", quality=50, optimize=True)
    os.remove(filename)

def worker():
    while True:
        try:
            hashtag, id, url = job_pool.get(timeout=10)
            print(url)
            get_image(url, f"./hashtag/{hashtag.lower()}/{id}.{url[-3:]}")
            job_pool.task_done()
        except queue.Empty:
            break

tweets = []

c = twint.Config()
c.Search = f"#{HASHTAG}"
c.Limit = NUM_TWEETS
c.Images = True
c.Min_likes = 2
c.Filter_retweets = True
c.Store_object = True
c.Store_object_tweets_list = tweets
c.Hide_output = True
# c.Output = f"{HASHTAG}.json"

twint.run.Search(c)

all_threads = [threading.Thread(target=worker) for _ in range(MAX_WORKERS)]
for t in all_threads:
    t.start()

print("-"*50 + "\nSending jobs to pool")
datafile.write("ID,PERMALINK,HASHTAG,IMG URL\n")
for twt in tweets:
    datafile.write(f"{twt.id},\"{twt.link}\",{HASHTAG},\"{twt.photos[0]}\"\n")
    job_pool.put((HASHTAG, twt.id, twt.photos[0]))

print("All jobs sent")
for i in range(20):
    time.sleep(5)
    size = job_pool.qsize()
    print(f"Pool size: {size}")
    if size == 0:
        break

job_pool.join()
print("Done!")
