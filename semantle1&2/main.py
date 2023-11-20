#Semantle 1

# import requests

# input_data = "asteroid"
# #asteroid:
# {'flag': 'gAAAAABlLvszdnAZ-de99eGWANqkiqv9c0EvqOHD4OM3KoRSKBWijsjHzcjIwmgWs1MXjJLe-hXj6i_pK8ofIGDobbsEszt-l8ucZDh6smi3CXPFo_WzP69ejSJ4ZvuqGYJDPStfC51OCDPjnlTu6IeNbnvMJvti-yliM_iJkt7cGwTx5dJfFfs='}
# #planet,pluto 0.87
# #jupiter,venus 0.86
# #geo,earth,stone,moon,rocket,satelite 0.84
# #rock 0.82
# #mantle,material,thing,space,diamond 0.81
# #object,tool = 0.8
# #invisible > visible
# #0.78
# def query(input_data):
#     response = requests.post('http://semantle.advml.com/score', json={'data': input_data})
#     return response.json()

# query(input_data)

###############################

#Semantle 2

# import requests
# #media, telecast, woman, man
# #vision, landscape, read, write, idea, thought, 
# input_data = [
#     "Person woman man camera TV."
# ]


#input_data = "woman girl woman man woman"

#2 !@#EAS - 34531 -  fart - deaad - art - cart - girl - woman
#2 84  -   85         86   - 87      88  - 89   - 90 - 91
#1 and 3 are farther from woman than the rest
#2 and #5 mediium
#4 man is closest to ground
#1 woman> man
#4 girl~

#"woman woman woman man woman" 0.91
#"object home alive human wildlife" 0.83
#1 object 0.81, not verb, visible, unusable? not physical
#2 not verb, place++,living? more home than city : dwelling < home, good test
#3 alive, not object, not solid or liquid, bird?
#4 not living? alive more friendly than dangerous, mammal, no fur? MAN - mammal
#5 not verb, action?living?animal+








# #Person woman man camera TV.{'flag': 'gAAAAABlMqcsK6Bk4nSzr7ywWiUmqkTG77zpkHoBd9_YJgn2kugTaLATyo86t93j44N5XiuiZsBb094uctcpigDgRXEj8xs7mTjOd4Zn0UGQFPm589jS2rCeRU2gQJ0FMlP8qW4A5SWG8GtXkp-93G3JV1-F3UsWsa9s3HhcTR1hH5Kyoe7TRVk='}



# #2 !@#EAS - 34531 -  fart - deaad - art - cart - girl - woman
# #2 84  -   85         86   - 87      88  - 89   - 90 - 91
# #1 and 3 are farther from woman than the rest
# #2 and #5 mediium
# #4 man is closest to ground
# #1 woman> man
# #4 girl~

# #"woman woman woman man woman" 0.91
# #"object home alive human wildlife" 0.83
# #1 object 0.81, not verb, visible, unusable? not physical
# #2 not verb, place++,living? more home than city : dwelling < home, good test
# #3 alive, not object, not solid or liquid, bird?
# #4 not living? alive more friendly than dangerous, mammal, no fur? MAN - mammal
# #5 not verb, action?living?animal+
# def query(input_data):
#     response = requests.post('http://semantle2.advml.com/score', json={'data': input_data})
#     return response.json()
# for i in input_data:
#     print(f"{i}{query(i)}")
# print(query(input_data))
# # woman woman woman man video{'message': 0.92}
# # woman woman woman man media{'message': 0.92}
# # woman woman woman man telecast{'message': 0.92}

# # woman woman girl man telecast{'message': 0.92}
# # woman woman girl man video{'message': 0.92}
# # woman woman girl man media{'message': 0.92}
# # girl woman woman man telecast{'message': 0.92}
# # woman girl woman man telecast{'message': 0.92}
# # woman girl woman man telecast{'message': 0.92}
# # woman man telecast woman girl{'message': 0.92}
# # girl woman man telecast woman{'message': 0.92}
# # woman girl man telecast woman{'message': 0.92}
# # girl woman man woman telecast{'message': 0.92}
# # woman man girl woman telecast{'message': 0.92}
# # woman girl man telecast woman{'message': 0.92}
# # woman man girl telecast woman{'message': 0.92}
# # woman girl man woman telecast{'message': 0.92}
# # girl woman man video telecast{'message': 0.92}
# # video man woman girl woman{'message': 0.92}
# # media woman man girl woman{'message': 0.92}
# # girl man woman telecast woman{'message': 0.92}
# # video woman woman man girl{'message': 0.92}
# # media man woman woman girl{'message': 0.92}
# # video woman girl man woman{'message': 0.92}
# # woman woman joy girl man telecast{'message': 0.91}
# # video table man woman girl woman{'message': 0.91}
# # woman woman city man telecast{'message': 0.91}
# # woman woman home man telecast{'message': 0.91}
# # video woman man city village{'message': 0.9}
# # video woman man idea concept{'message': 0.9}
# # video woman man speak listen{'message': 0.9}
# # woman man read write telecast{'message': 0.89}
# # man woman ponder reflect telecast{'message': 0.89}
# # video woman man question answer{'message': 0.89}

# # video man woman town country{'message': 0.9}
# # video man woman country city{'message': 0.9}
# # video man woman talk discuss{'message': 0.89}
# # video woman man nature technology{'message': 0.91}
# # video woman man ancient modern{'message': 0.89}
# # video woman man earth space{'message': 0.9}
# # video man woman nature tech{'message': 0.9}
# # camera woman man deliberate laptop{'message': 0.9}
# # television man woman speak technology{'message': 0.9}
# # video tech man woman nature{'message': 0.91}
# # video technology woman man nature{'message': 0.91}
# # video woman man landscape device{'message': 0.91}
# # video woman man sky satellite{'message': 0.9}
# # video woman man habitat computer{'message': 0.9}
# # video woman man machine human{'message': 0.91}
# # video woman man speak talk{'message': 0.91}
# # video man woman talk speak{'message': 0.91}
# # man woman form compose camera{'message': 0.91}
# # woman man technology universe phonecamera{'message': 0.9}
# # man woman form write camera{'message': 0.92}
# # man woman write camera video{'message': 0.92}
# # video man woman write camera{'message': 0.93}

# # woman man document camera{'message': 0.92}
# # woman man form camera{'message': 0.92}
# # video woman man synthesize camera{'message': 0.92}
# # video man woman write camera{'message': 0.93}
# # video man woman takes camera{'message': 0.93}
# # video man woman films camera{'message': 0.93}
# # video man woman with camera{'message': 0.94}
# # video man woman of camera{'message': 0.94}
# # video man woman too camera{'message': 0.94}
# # video man woman by camera{'message': 0.94}
