import requests

import time
# Function to fetch word list from the URL
def fetch_word_list():
    url = "https://raw.githubusercontent.com/Taknok/French-Wordlist/master/francais.txt"
    response = requests.get(url)
    return response.text.splitlines()

# Fetch the word list
words = fetch_word_list()
print(words[:10])
import sqlite3

# 1. Create a simple SQLite database
def init_db():
    """Initialize the SQLite database and create a table if it doesn't exist."""
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS api_cache (word TEXT PRIMARY KEY, response TEXT)''')
    conn.commit()
    conn.close()

init_db()

# 2. Implement functions to interact with the database
def get_from_db(word):
    """Get the response from the database for the given word."""
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute('SELECT response FROM api_cache WHERE word = ?', (word,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

def insert_into_db(word, response):
    """Insert a word and its corresponding response into the database."""
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO api_cache (word, response) VALUES (?, ?)', (word, response))
    conn.commit()
    conn.close()

# Filter words based on the criteria
filtered_words = [word for word in words if len(word) == 8 and word[5] == 't' and word[6] == 'a']# and word[6] == 'a']# and word[0] == 'f']

# Query function
def query(data):
    """Query the API with the given data or return from cache if available."""
    # Check if response is in the database
    cached_response = get_from_db(data)
    # print(cached_response)
    if cached_response:
        return cached_response
    
    # If not in database, query the API
    response = requests.post('http://inversion.advml.com/score', json={'flag': data})
    response_data = response.json()
    
    # Insert the response into the database
    insert_into_db(data, str(response_data))
    
    return response_data

# Query each word and print results
# for word in filtered_words:
#     result = query(word)
#     print(f"Result for {word}: {result}")



first_list = ['l']
second_list = ['e']
third_list = ['t','+']
fourth_list = ['m', 'n', 'h']
fifth_list = list('aeiou')
sixth_list = list('abcdefghijklmnopqrstuvwxyz')
seventh_list = ['u']
eighth_list = list('abcdefghijklmnopqrstuvwxyz')
filtered_words = []
btrx = 0 # to 9000
it = 0
#
#14332
for word in filtered_words:
    result = query(word)
    print(f"Result for {word}: {result}")
for first in first_list:
    for second in second_list:
        for third in third_list:
            for fourth in fourth_list:
                for fifth in fifth_list:
                    for sixth in sixth_list:
                        for seventh in seventh_list:
                            for eighth in eighth_list:
                                word = first + second + third + fourth + fifth + sixth + seventh + eighth
                                success = False
                                it += 1
                                if it < btrx:
                                  continue
                                while success != True:
                                    try:
                                        result = query(word)
                                        if result:
                                            print(f"{it} : Result for  {word}: {result}")
                                            success = True
                                            break
                                    except:
                                        print("Sleeping")
                                        time.sleep(31)
