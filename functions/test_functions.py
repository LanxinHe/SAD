import numpy as np


def gray_map(rate):
    # table = np.zeros([np.power(2, rate), rate])
    x = np.array([[0], [1]], dtype='int')     # initial mapping for 1 bit
    for m in range(rate-1):
        length_of_x = x.shape[0]
        upart = np.hstack([np.zeros(([length_of_x, 1])), x])
        dpart = np.hstack([np.ones([length_of_x, 1]), np.flipud(x)])
        x = np.vstack([upart, dpart])
    table = x.astype(np.int)
    return table


def gray_ber(prediction, data_real, data_imag, rate=2):

    K = prediction.shape[0]
    tx = round(prediction.shape[1]/2)
    l = 2 ** rate
    table = gray_map(rate)
    ber = 0
    for k in range(K):
        for t in range(2*tx):
            if prediction[k, t] < 0.:
                prediction[k, t] = 0.
            elif prediction[k, t] > (l-1):
                prediction[k, t] = l-1
    prediction = np.transpose(prediction)
    for k in range(K):
        for t in range(tx):
            ber = ber +\
                np.sum(np.bitwise_xor(table[np.int(data_real[t, k]), :], table[np.int(prediction[t, k]), :])) +\
                np.sum(np.bitwise_xor(table[np.int(data_imag[t, k]), :], table[np.int(prediction[t+tx, k]), :]))
    ber = ber / K / tx / rate / 2
    return ber