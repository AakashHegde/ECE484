import matplotlib.pyplot as plt
import json
#import csv

#with open('.csv', newline='') as csvfile:
#    reader = csv.Di

time = []
distance = []
heading = []

with open('plotD.json') as f:
    data = json.load(f)
    #for i in data[]

for key, value in data.items():
    time.append(float(key))
    distance.append(value[0])
    heading.append(value[1])


plt.plot(time, heading)
#plt.xlabel('Time')
#plt.ylabel('Distance Error')
#plt.title('Distance Error vs Time')

plt.ylabel('Heading Error')
plt.title('Heading Error vs Time')

plt.show()