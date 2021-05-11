import time
import random
import requests
import pandas as pd


def save_data(data, path):
    df = pd.DataFrame(data)
    df.to_json(path, indent=4)

def reddit_scraper(url, number_of_scrapes, path):
    
    after = None 
    params = {}
    output_list = []
    headers = {"User-agent" : "Gh Mhdi"}

    print("start proccess : ", url)

    for i in range(number_of_scrapes):
       
        if i % 5 == 0:print("Downloading Batch" , i, "of", number_of_scrapes)
        
        res = requests.get(url, params=params, headers=headers)
        if res.status_code == 200:
            the_json = res.json()
            output_list.extend(the_json["data"]["children"])
            after = the_json["data"]["after"]

        #THIS WILL TELL THE SCRAPER TO GET THE NEXT SET AFTER REDDIT'S after CODE
        params = {"after": after}             
        
        #sleep between request
        time.sleep(random.randint(1,6))
    
    post_name = set([p["data"]["name"] for p in output_list])
    output_list = [p["data"] for p in output_list if p["data"]["name"] in post_name]
    
    save_data(output_list, path)
    

def main():
    subreddit=[['https://www.reddit.com/r/depression.json', '0'] ,['https://www.reddit.com/r/stress.json', '0'], ['https://www.reddit.com/r/happiness.json', '1'], ['https://www.reddit.com/r/success.json', '1']]

    for url, label in subreddit:
        path = '../data/row/'+ label + '/' + url.replace('https://www.reddit.com/r/', '') 
        reddit_scraper(url, 5, path)


if __name__ == '__main__':
    main()
