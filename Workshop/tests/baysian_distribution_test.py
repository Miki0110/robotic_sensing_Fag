import cv2 as cv
import numpy as np

features = ["holes", "circularity", "compactness", "elongation", "intensity"]
# Probability stats
mu1 = np.array([(1.2667, 0.1604, 0.4056, 0.6069, 0.4252)])
mu2 = np.array([(0.6533, 0.2973, 0.4741, 0.7961, 0.4513)])
mu3 = np.array([(0.6533, 0.2973, 0.4741, 0.7961, 0.4513)])
mu4 = np.array([(11.4467, 0.0682, 0.5006, 1.5439, 0.5659)])

detsigma1 = 7.07165790821967e-12
detsigma2 = 1.13458722970304e-11
detsigma3 = 1.93738742118705e-09
detsigma4 = 6.72977393610626e-08

invsigma1 = np.array([(0.776534671018742,	24.8928266241895,	-19.4694011403276,	-17.2238003733083,	-3.85802336585192),
                      (24.8928266241895,	3116.24896002362,	-2481.12562649331,	-567.713478109552,	47.5858504378984),
                      (-19.4694011403276,	-2481.12562649331,	4056.06731596908,	358.713620347677,	-103.660225286553),
                    (-17.2238003733083, -567.713478109552,	358.713620347677,	983.587924756339,	175.038412438643),
                    (-3.85802336585192,	47.5858504378984,	-103.660225286553, 175.038412438643,	109.842698000308)])
invsigma2 = np.array([[2.01934690607180, 38.8085918105957, -62.2962810695715,	-2.04512308503161,	-0.281526939730740],
                    [38.8085918105957,	4243.08060193304, -4297.09399705085,	-843.484487017106,	-68.1796201193774],
                    [-62.2962810695715,	-4297.09399705085, 7501.94434652172,	-3.32666263039580,	27.4383910782106],
                    [-2.04512308503161,	-843.484487017106, -3.32666263039580,	497.943720393071,	19.3414878087448],
                    [-0.281526939730740,-68.1796201193774, 27.4383910782106,	19.3414878087448,	47.5794905891147]])
invsigma3 = np.array([[0.548097016464341,	21.1922998832028,	-5.59636250917097,	-1.22445747609375,	4.34303212721654],
                    [21.1922998832028,	1442.80654031399,	-527.194193323870,	-100.576351373209,	348.421265075554],
                    [-5.59636250917097,	-527.194193323870,	573.742905572009,	-10.2451745759608,	-208.244369251573],
                    [-1.22445747609375,	-100.576351373209, -10.2451745759608,	48.4140866441447,	29.3574895722035],
                    [4.34303212721654,	348.421265075554, -208.244369251573,	29.3574895722035,	280.431998512421]])
invsigma4 = np.array([[0.0247628029967340,	2.38444365707769,	-0.249377592581288,	-0.384200235861934,	-0.754236353289016],
                    [2.38444365707769,	2574.03081499479,	-541.277328498655, -59.6003666292376,	-63.9602257532718],
                    [-0.249377592581288, -541.277328498655, 371.477419224037,	-39.6244042693563,	-29.5938339939882],
                    [-0.384200235861934, -59.6003666292376,	-39.6244042693563,	31.8098843564715,	26.2551035174680],
                    [-0.754236353289016, -63.9602257532718, -29.5938339939882,	26.2551035174680,	92.9295774670917]])

def check_chance(x):
    g1=1/np.sqrt(2*np.pi*detsigma1)*np.exp(-0.5*(x-mu1).dot(invsigma1.dot((x-mu1).T)))
    g2=1/np.sqrt(2*np.pi*detsigma2)*np.exp(-0.5*(x-mu2).dot(invsigma2.dot((x-mu2).T)))
    g3=1/np.sqrt(2*np.pi*detsigma3)*np.exp(-0.5*(x-mu3).dot(invsigma3.dot((x-mu3).T)))
    g4=1/np.sqrt(2*np.pi*detsigma4)*np.exp(-0.5*(x-mu4).dot(invsigma4.dot((x-mu4).T)))
    return(g1[0][0],g2[0][0],g3[0][0],g4[0][0])

bass = np.array([(0.2000,0.1809,0.4241,0.5842,0.3519)])
guitar = np.array([(0,0.3041,0.4731,0.8228,0.3214)])
trumpet = np.array([(1.9000,0.2373,0.5231,0.7469,0.5975)])
drumm = np.array([(10.7000,0.0394,0.4263,1.560,0.5735)])
print(check_chance(bass))
print(check_chance(guitar))
print(check_chance(trumpet))
print(check_chance(drumm))