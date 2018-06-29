import numpy as np

supportedModulation = ["2pam", "4pam", "8pam", "2psk", "4psk", "8psk", "4qam", "16qam", "64qam", "256qam"]

def getsymbol(modulationType):
    "GETALPHABET    Generate set of alphabet according to modulation type."
    # Create basic mapping of signal symbols
    if modulationType == '2pam':
        symbolMap = np.array([1, -1])
    elif modulationType == '4pam':
        symbolMap = np.array([-3, -1, 1, 3])
    elif modulationType == '8pam':
        symbolMap = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    elif modulationType == '2psk':
        symbolMap = np.array([1j, -1j])
    elif modulationType == '4psk':
        symbolMap = np.array([1, 1j, -1, -1j])
    elif modulationType == '8psk':
        symbolMap = np.array([np.sqrt(2), 1+1j, np.sqrt(2)*1j, -1+1j,\
                            -np.sqrt(2), -1-1j, -np.sqrt(2)*1j, 1-1j])
    elif modulationType == '4qam':
        symbolMap = np.array([1+1j, -1+1j, -1-1j, 1-1j])
    elif modulationType == '16qam':
        symbolMap = np.array([3+3j, 3+1j, 3-1j, 3-3j, 1+3j, 1+1j, 1-1j, 1-3j,\
                -1+3j, -1+1j, -1-1j, -1-3j, -3+3j, -3+1j, -3-1j, -3-3j])
    elif modulationType == '64qam':
        symbolMap = np.array([1+1j, 3+1j, 1+3j, 3+3j, 7+1j, 5+1j, 7+3j,\
                5+3j, 1+7j, 3+7j, 1+5j, 3+5j, 7+7j, 5+7j, 7+5j, 5+5j, 1-1j,\
                1-3j, 3-1j, 3-3j, 1-7j, 1-5j, 3-7j, 3-5j, 7-1j, 7-3j, 5-1j,\
                5-3j, 7-7j, 7-5j, 5-7j, 5-5j, -1+1j, -1+3j, -3+1j, -3+3j,\
                -1+7j, -1+5j, -3+7j, -3+5j, -7+1j, -7+3j, -5+1j, -5+3j,\
                -7+7j, -7+5j, -5+7j, -5+5j, -1-1j, -3-1j, -1-3j, -3-3j,\
                -7-1j, -5-1j, -7-3j, -5-3j, -1-7j, -3-7j, -1-5j, -3-5j,\
                -7-7j, -5-7j, -7-5j, -5-5j])
    elif modulationType == '256qam':
        symbolMap = np.array([1+1j, 1+3j, 1+5j, 1+7j, 1+9j, 1+11j, 1+13j,\
                1+15j, 1-1j, 1-3j, 1-5j, 1-7j, 1-9j, 1-11j, 1-13j, 1-15j,\
                3+1j, 3+3j, 3+5j, 3+7j, 3+9j, 3+11j, 3+13j, 3+15j, 3-1j,\
                3-3j, 3-5j, 3-7j, 3-9j, 3-11j, 3-13j, 3-15j, 5+1j, 5+3j,\
                5+5j, 5+7j, 5+9j, 5+11j, 5+13j, 5+15j, 5-1j, 5-3j, 5-5j,\
                5-7j, 5-9j, 5-11j, 5-13j, 5-15j, 7+1j, 7+3j, 7+5j, 7+7j,\
                7+9j, 7+11j, 7+13j, 7+15j, 7-1j, 7-3j, 7-5j, 7-7j, 7-9j,\
                7-11j, 7-13j, 7-15j, 9+1j, 9+3j, 9+5j, 9+7j, 9+9j, 9+11j,\
                9+13j, 9+15j, 9-1j, 9-3j, 9-5j, 9-7j, 9-9j, 9-11j, 9-13j,\
                9-15j, 11+1j, 11+3j, 11+5j, 11+7j, 11+9j, 11+11j, 11+13j,\
                11+15j, 11-1j, 11-3j, 11-5j, 11-7j, 11-9j, 11-11j, 11-13j,\
                11-15j, 13+1j, 13+3j, 13+5j, 13+7j, 13+9j, 13+11j, 13+13j,\
                13+15j, 13-1j, 13-3j, 13-5j, 13-7j, 13-9j, 13-11j, 13-13j,\
                13-15j, 15+1j, 15+3j, 15+5j, 15+7j, 15+9j, 15+11j, 15+13j,\
                15+15j, 15-1j, 15-3j, 15-5j, 15-7j, 15-9j, 15-11j, 15-13j,\
                15-15j, -1+1j, -1+3j, -1+5j, -1+7j, -1+9j, -1+11j, -1+13j,\
                -1+15j, -1-1j, -1-3j, -1-5j, -1-7j, -1-9j, -1-11j, -1-13j,\
                -1-15j, -3+1j, -3+3j, -3+5j, -3+7j, -3+9j, -3+11j, -3+13j,\
                -3+15j, -3-1j, -3-3j, -3-5j, -3-7j, -3-9j, -3-11j, -3-13j,\
                -3-15j, -5+1j, -5+3j, -5+5j, -5+7j, -5+9j, -5+11j, -5+13j,\
                -5+15j, -5-1j, -5-3j, -5-5j, -5-7j, -5-9j, -5-11j, -5-13j,\
                -5-15j, -7+1j, -7+3j, -7+5j, -7+7j, -7+9j, -7+11j, -7+13j,\
                -7+15j, -7-1j, -7-3j, -7-5j, -7-7j, -7-9j, -7-11j, -7-13j,\
                -7-15j, -9+1j, -9+3j, -9+5j, -9+7j, -9+9j, -9+11j, -9+13j,\
                -9+15j, -9-1j, -9-3j, -9-5j, -9-7j, -9-9j, -9-11j, -9-13j,\
                -9-15j, -11+1j, -11+3j, -11+5j, -11+7j, -11+9j, -11+11j,\
                -11+13j, -11+15j, -11-1j, -11-3j, -11-5j, -11-7j, -11-9j,\
                -11-11j, -11-13j, -11-15j, -13+1j, -13+3j, -13+5j, -13+7j,\
                -13+9j, -13+11j, -13+13j, -13+15j, -13-1j, -13-3j, -13-5j,\
                -13-7j, -13-9j, -13-11j, -13-13j, -13-15j, -15+1j, -15+3j,\
                -15+5j, -15+7j, -15+9j, -15+11j, -15+13j, -15+15j, -15-1j,\
                -15-3j, -15-5j, -15-7j, -15-9j, -15-11j, -15-13j, -15-15j])

    symbolMap = symbolMap/np.sqrt(np.mean(np.power(np.abs(symbolMap),2)))
    return symbolMap

def genmodsig(modulationType,sampleNo):
    "GENMODSIG   Generate i.i.d modulated signal samples with unit power"
    # Create basic mapping of signal symbols
    symbolMap = getsymbol(modulationType)

    # Calculate signal power
    symbolPower = np.mean(np.square(np.absolute(symbolMap)))

    # Normalise symbol
    symbolMap = np.divide(symbolMap,np.sqrt(symbolPower))

    # Create uniform random index of signal samples
    symbol = np.floor(np.random.rand(sampleNo)*symbolMap.size)
    symbol = symbol.astype(int)

    sigOut = symbolMap[symbol]
    return sigOut
