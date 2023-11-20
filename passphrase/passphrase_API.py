import requests
import time
import json
import re

# My secret sauce is the best!
target = {'scores': {'negative': 0.0022, 'neutral': 0.0325, 'positive': 0.9653}}


def compute_mse(given_scores, api_response_scores):
    mse = (
                  (given_scores['negative'] - api_response_scores['negative']) ** 2 +
                  (given_scores['neutral'] - api_response_scores['neutral']) ** 2 +
                  (given_scores['positive'] - api_response_scores['positive']) ** 2
          ) / 3
    return mse


with open('index.json', 'r') as file:
    data = json.load(file)
with open('nouns.txt', 'r') as file:
    # Read lines and strip any whitespace
    data = [line.strip() for line in file]


def query(input_data):
    response = requests.post('http://passphrase.advml.com/score', json={'data': input_data})
    return response.json()


# Filter names based on the given conditions
filtered_names = [name for name in data if re.match("^[A-Za-z]+$", name) and len(name) >= 1]
print(len(filtered_names))
it = 0
btrx = 0
best = 1
for name in reversed(filtered_names):
    it += 1
    if it < btrx or len(name) > 8:
        continue
    res = f"The {name} visual is also immensely appealing."
    # print(res)
    # print(len(name))
    s = 0
    while s != 1:
        try:
            ret = query(res)
            print(f"{it} : {name} : {ret}")
            mse = compute_mse(target["scores"], ret["scores"])

            if mse <= best:
                best = mse
                print(f"\n\nNew best at {it} with MSE = {1 / best} with string {res}\n\n")
            s = 1
        except:
            print("Sleeping!")
            time.sleep(11)



# tried:
# f"The {name} visuals are also immensely appealing"
# f"The {name} visuals are also immensely appealing."


#https://chat.openai.com/c/0fadc1cc-968d-4618-bc10-fd6ee0eb00b0
