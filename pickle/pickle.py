# import requests
# import base64
# import pickle
# import os


# def a():
#     return eval
# #     "{'flag': 'gAAAAABlL_zia0I9q0VXQkfjGHmrH4Gz5yxzqOkz5ST8O6Tkspm_wkMv7bZ133uCMsXMu37uud4BVLC4_q6pIFeJz-RttvP-2auBsBUM07Jzkc22ihfsq1QPUoejqPz3Fo1lUpTE2B0YBm9vtKbgzO4szf1C4javxGlw_W0UtU8HOksLX785NXg='}"
# # dangerous_pickle = pickle.dumps(Exploit())

# obj = pickle.dumps(a())
# input_data = base64.b64encode(obj).decode()

# def query(input_data):
#     response = requests.post("http://pickle.advml.com/score", json={"data": input_data})
#     return response.json()

# query(input_data)
