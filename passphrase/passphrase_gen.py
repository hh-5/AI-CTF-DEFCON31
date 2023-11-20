# import nltk
import requests
import sqlite3
import markovify
import time
# from nltk.corpus import brown, movie_reviews, reuters
# print(brown.categories())
# nltk.download('movie_reviews')
#
# nltk.download('punkt')
# print(movie_reviews.categories())
# print(reuters.categories())
# 1. Load seed sentences for sentence generation

with open("seed_sentences.txt", "r") as f:
    seed_text = f.read()

# 2. Sentence Generation
print(len(seed_text))
text_model = markovify.Text(seed_text, state_size=2)

print(len(seed_text))

# 3. SQLite Database Setup and Storage
conn = sqlite3.connect("sentiment_data.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS sentiment_data (
    sentence TEXT PRIMARY KEY,
    positive_score REAL,
    neutral_score REAL,
    negative_score REAL
)
""")

# 4. Function to query the API
def query(input_data, conn):
    # First, check if the sentence is already in the database
    cursor = conn.cursor()
    cursor.execute("SELECT positive_score, neutral_score, negative_score FROM sentiment_data WHERE sentence=?",
                   (input_data,))
    result = cursor.fetchone()

    # If the sentence is found in the database, return the scores
    if result:
        return {
            'scores': {
                'positive': result[0],
                'neutral': result[1],
                'negative': result[2]
            }
        }
    # If the sentence is not in the database, make the API call
    else:
        s = False
        # print("Got here")
        while s != True:
            try:
                response = requests.post('http://passphrase.advml.com/score', json={'data': input_data})
                if response:
                    s = True
            except:
                time.sleep(31)
                print("sleeping")

        print(f"{input_data} : {response.json()}")
        sentiment_scores = response.json()

        # Store the result in the database
        cursor.execute("""
        INSERT INTO sentiment_data (sentence, positive_score, neutral_score, negative_score)
        VALUES (?, ?, ?, ?)
        """, (input_data, sentiment_scores['scores']['positive'], sentiment_scores['scores']['neutral'],
              sentiment_scores['scores']['negative']))

        conn.commit()

        return sentiment_scores

# 5. Main loop to generate sentences, query sentiments, and store in the database
while True:
    conn = sqlite3.connect("sentiment_data.db")
    cursor = conn.cursor()
    generated_sentences = set()

    while len(generated_sentences) < 10:
        sentence = text_model.make_short_sentence(max_chars=50)
        # print(len(generated_sentences))
        if sentence:
            generated_sentences.add(sentence)

    sentiment_results = [(sentence, query(sentence,conn)) for sentence in generated_sentences]

    # Insert data into the table
    for sentence, result in sentiment_results:
        scores = result['scores']
        cursor.execute("""
        INSERT OR IGNORE INTO sentiment_data (sentence, positive_score, neutral_score, negative_score)
        VALUES (?, ?, ?, ?)
        """, (sentence, scores['positive'], scores['neutral'], scores['negative']))

    conn.commit()
    conn.close()
