import numpy as np
import matplotlib.pyplot as plt

def blazed_grating_diffraction(wavelength, slit_width, grating_period, grating_angle, observation_angle):
    """
    Calculate the intensity of light at a given observation angle due to a blazed grating.

    :param wavelength: Wavelength of the incident light (in meters)
    :param slit_width: Width of the slit (in meters)
    :param grating_period: Period of the grating (in meters)
    :param grating_angle: Blaze angle of the grating (in degrees)
    :param observation_angle: Observation angle (in degrees)
    :return: Intensity at the observation angle
    """
    # Convert angles from degrees to radians
    grating_angle_rad = np.radians(grating_angle)
    observation_angle_rad = np.radians(observation_angle)

    # Calculate the diffraction order using the grating equation
    m = (wavelength / grating_period) * (np.sin(grating_angle_rad) + np.sin(observation_angle_rad))

    # Intensity calculation for single-slit diffraction
    alpha = (np.pi * slit_width / wavelength) * np.sin(observation_angle_rad)
    if alpha == 0:
        intensity = 1  # To avoid division by zero
    else:
        intensity = (np.sin(alpha) / alpha) ** 2

    return intensity, m

# Parameters
wavelength = 500e-9  # 500 nm
slit_width = 1e-5    # 10 micron
grating_period = 3.5e-6  # 3.5 micron
grating_angle = 8    # 10 degrees

# Observation angles
observation_angles = np.linspace(-90, 90, 500)
intensities = np.array([blazed_grating_diffraction(wavelength, slit_width, grating_period, grating_angle, angle)[0] for angle in observation_angles])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(observation_angles, intensities, label='Diffraction Intensity')
plt.xlabel('Observation Angle (degrees)')
plt.ylabel('Intensity (arbitrary units)')
plt.title('Diffraction Pattern of a Blazed Grating')
plt.legend()
plt.grid(True)
plt.show()
