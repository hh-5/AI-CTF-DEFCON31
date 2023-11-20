For passphrase, I omitted the approaches that did not yield useful results.

The process used here was to generate a set of short markovified sentences and evaluate them. The seeds are in seed_sentences.txt and the results are in sentiment_data.db.

The best-scoring sentence from the db was used to get the flag by varying the initial part of the sentence + adding a noun from nouns.txt. 

Although using names instead of nouns leads to the correct results without giving the flag, I did include the name set I used (index.json). Out of the correct variations, while "Ancog" was an odd name, "Week" was both a name and a noun - and based on my understanding at the time it should have worked.
