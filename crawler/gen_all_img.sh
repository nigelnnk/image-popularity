#!/bin/bash
NUM_FILES=550
conda activate reddit
sleep 1

echo "" > log.txt

input="./subreddit_list.txt"
while IFS= read -r line
do
  line=$(echo "$line" | tr '[:upper:]' '[:lower:]')
  echo "$(date +%H:%M) Crawling from /r/$line"
  python crawler.py -s "$line" -i "$NUM_FILES" >> log.txt
  files=$(ls "$line/" | wc -l)
  echo "$(date +%H:%M) Done getting $files/$NUM_FILES. Space: $(du -sh "$line/")"
done < "$input"

echo "All done!"
