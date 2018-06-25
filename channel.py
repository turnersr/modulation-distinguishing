import numpy as np
def awgn(sigIn,snr):
    "AWGN   Adds AWGN noise to signals according to Signal-to-noise ratio."

    # Determine the dimension of input signal data
    N = sigIn.size

    # Generate unscaled AWGN noise components
    noise = np.random.randn(N) + np.random.randn(N)*1j

    # Calculate signal power
    sigPower = np.mean(np.square(np.absolute(sigIn)))

    # Calculate noise power
    noisePower = np.mean(np.square(np.absolute(noise)))

    # Calculate scaling factor for noise components according to SNR
    sigNoise = np.sqrt(sigPower/noisePower)*(np.power(10,-snr/20.0))

    # Map the noise power scaling factor
    sigOut = sigIn + noise*sigNoise

    return sigOut
