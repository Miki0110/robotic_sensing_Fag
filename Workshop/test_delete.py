import numpy as np
import math
import json

instruments = ['guitar', 'bass', 'trumpet']
feature_data = ['circularity', 'compactness', 'elongation', 'thiness', 'intensity']
test_data = [0.3, 0.4, 0.2, 0.5, 0.3]
instrument_data = []
for instrument in instruments:
    instrument_data.append(json.load(open(f'data_{instrument}.json', 'r')))

guitar_distance = 0
bass_distance = 0
trumpet_distance = 0

for i in range(len(feature_data)):
    guitar_distance += pow(test_data[i]-np.array(instrument_data[0][feature_data[i]]), 2)
    bass_distance += pow(test_data[i]-np.array(instrument_data[1][feature_data[i]]), 2)
    trumpet_distance += pow(test_data[i] - np.array(instrument_data[2][feature_data[i]]), 2)

guitar_distance = min(np.sqrt(guitar_distance))
bass_distance = min(np.sqrt(bass_distance))
trumpet_distance = min(np.sqrt(trumpet_distance))

if guitar_distance > bass_distance and guitar_distance > trumpet_distance:
    print("guitar")
elif bass_distance > guitar_distance and bass_distance > trumpet_distance:
    print("bass")
else:
    print("trumpet")


