import datetime
import os
import praw
import queue
import requests
import threading

from argparse import ArgumentParser
from PIL import Image

def make_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--max_imgs', dest='max_img', default=600, type=int,
                        help="Number of images to download")
    parser.add_argument('-w', '--num_workers', dest='no_workers', default=6, type=int,
                        help="Number of worker threads to run with")
    parser.add_argument('-s', '--subreddit', dest='sr', default="earthporn", type=str,
                        help="Name of subreddit to download images from")
    parser.add_argument('-r', '--resolution', dest='max_reso', default=1024, type=int,
                        help="Maximum resolution of image")
    args = parser.parse_args()
    return args

args = make_args()
SUBREDDIT = args.sr
MAX_IMG = args.max_img
MAX_WORKERS = args.no_workers
MAX_RESOLUTION = (args.max_reso, args.max_reso)
ACCEPTABLE_EXTENSIONS = ["jpg", "png"]
reddit = praw.Reddit("cs4243")
job_pool = queue.Queue()

def get_image(url, filename) :
    req = requests.get(url, stream=True)
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
        subr, id, url = job_pool.get()
        get_image(url, f"./{subr.lower()}/{id}.{url[-3:]}")
        job_pool.task_done()
        if job_pool.empty():
            break

all_threads = [threading.Thread(target=worker) for _ in range(MAX_WORKERS)]
for t in all_threads:
    t.start()

now = datetime.datetime.now()
unixnow = int(datetime.datetime.timestamp(now))
datafile = open(f"crawl_{now.month:02}{now.day:02}_{SUBREDDIT}.csv", 'w')
datafile.write("CS4243_crawler data\n")
datafile.write(f"subreddit: {SUBREDDIT} \n")
os.makedirs(f"./{SUBREDDIT.lower()}", exist_ok=True)

all_count = 0
all_limit = MAX_IMG
datafile.write("SUBREDDIT,ID,SCORE,URL,\n")
for submission in reddit.subreddit(SUBREDDIT).new(limit=None):
    if (unixnow - submission.created_utc) > 604800:
        srname = submission.subreddit.display_name.lower()
        if submission.url[-3:] not in ACCEPTABLE_EXTENSIONS:
            continue
        datafile.write(f"{srname},{submission.id},{submission.score},{submission.url}\n")
        job_pool.put((srname, submission.id, submission.url))
        all_count += 1

        if all_count % 10 == 0:
            datafile.flush()
            print(f"Sourcing {all_count}/{all_limit}")
        if all_count == all_limit:
            break
job_pool.join()
print("Done!")
