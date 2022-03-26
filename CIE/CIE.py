    5import numpy as np
import colour

def Covert(array):
    res_dict = {array[i,0]: array[i,1] for i in range(0, len(array),2)}
    return res_dict

Spectro_data = np.genfromtxt('Control.csv', usecols=(0, 1), delimiter=',', skip_header=1)
sd = colour.SpectralDistribution(Covert(Spectro_data), name="Sample")
colour.plotting.plot_single_sd(sd) 
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
illuminant = colour.SDS_ILLUMINANTS['D65']

# Calculating the sample spectral distribution *CIE XYZ* tristimulus values.
XYZ = colour.sd_to_XYZ(sd, cmfs, illuminant)
print("XYZ: ",XYZ)
RGB = colour.XYZ_to_sRGB(XYZ / 100)
print("RGB: ",RGB)

colour.plotting.plot_single_colour_swatch(
    colour_swatch=RGB,
    text_kwargs={'size': 'x-large'})

"""
def XYZ_to_XY(X,Y,Z):
    return (X/(X+Y+Z), Y/(X+Y+Z))

Spectro_data = np.genfromtxt('Control.csv', usecols=(0, 1), delimiter=',', skip_header=1)
CIE_CMF = np.recfromtxt('cie-cmf.txt', delimiter=' ')
#Get File Data
print(type(Spectro_data[0]))

Spectro_data_wavelength = []
for ele in Spectro_data:
    Spectro_data_wavelength.append(ele[0])
print(Spectro_data_wavelength)


CalX = 0 ;CalY = 0;CalZ = 0
#set up X, Y, Z sum parameter

for row in CIE_CMF:
    if Spectro_data_wavelength[-1] > row[0] > Spectro_data_wavelength[0] :
        Index = Spectro_data_wavelength.index(row[0])
        CalX += Spectro_data[Index][1] * row[1]
        CalY += Spectro_data[Index][1] * row[2]
        CalZ += Spectro_data[Index][1] * row[3]
Cord = XYZ_to_XY(CalX, CalY, CalZ)
print(Cord)
"""