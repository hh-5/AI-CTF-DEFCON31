import requests
import time
# input_data = #(raw printed results here as list, can't reproduce)
def query(input_data):
    response = requests.post('http://cluster1.advml.com/score', json={'data': input_data})
    return response.json()
print(query(input_data))
# i = 0
# while i < 37000:
#     next_10 = [i+j for j in range(11)]
#     res = query(input_data + next_10)
#     if res['s'] > best:
#         for val in next_10:
#             if query(input_data + [val])['s'] > best:
#                 print(val)
#                 i = val + 1
#         i += 1
#     else:
#         i += 10
#     time.sleep(0.3)
#gAAAAABlM9GFCNe17hv31180YrUjxukKgOkSbyhEdwAfg9g306FBVZ_UXVn-0s20s2Yxc8W6zvfmWS09l_rdc_j7A0ejVNcDYuyKv69bIfI5m_yflZKumMe7O-taxPWQoTaqCqnAAnQBWw68_Mn1QfBoNn5tq1LFaFoa_TihZSm8JGpj7ZjFkR8='
