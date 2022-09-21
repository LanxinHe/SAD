import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder


def get_data(tx=8, rx=8, K=20000, rate=1, EbN0=15):
    """
    The data are generated with noise Eb/N0 = EbN0 dB
    """
    l = np.power(2, rate)    # PAM alphabet size
    # gray_table = gray_map(rate)

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])

    h_real = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_imag = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_com = np.vstack([np.hstack([h_real, -1*h_imag]),  # (2rx, 2tx, K)
                      np.hstack([h_imag, h_real])])

    noise_real = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_imag = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_com = np.hstack([noise_real, noise_imag])     # (K, 2rx)
    # y(K, 2rx)
    receive_signal = np.matmul(np.transpose(h_com, [2, 0, 1]), np.expand_dims(data_com.T, axis=-1)).squeeze(-1)

    snr = np.power(10, EbN0/10) * 3 / 2 / (l*l - 1) / tx * rate * 2
    var = 1 / snr
    std = np.sqrt(var)
    receive_signal_noised = receive_signal + std * noise_com    # (K, 2rx)

    # hty(K, 2tx), hth(K, 2tx, 2tx)
    # hty = np.matmul(np.transpose(h_com, [2, 1, 0]), np.expand_dims(receive_signal_noised, axis=-1)).squeeze(-1)
    # hth = np.matmul(np.transpose(h_com, [2, 1, 0]), np.transpose(h_com, [2, 0, 1]))

    return receive_signal_noised, np.transpose(h_com, [2, 0, 1]), data_real, data_imag


def get_wide_data(tx=8, rx=8, K=20000, rate=1):
    """
    The data are generated with the Eb/N0 about noise between [-1, 16] dB
    """
    l = np.power(2, rate)    # PAM alphabet size
    # gray_table = gray_map(rate)
    EbN0 = np.random.uniform(-1, 16, [K, 1])

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])

    h_real = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_imag = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_com = np.vstack([np.hstack([h_real, -1*h_imag]),  # (2rx, 2tx, K)
                      np.hstack([h_imag, h_real])])

    noise_real = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_imag = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_com = np.hstack([noise_real, noise_imag])     # (K, 2rx)
    # y(K, 2rx)
    receive_signal = np.matmul(np.transpose(h_com, [2, 0, 1]), np.expand_dims(data_com.T, axis=-1)).squeeze(-1)

    snr = np.power(10, EbN0/10) * 3 / 2 / (l*l - 1) / tx * rate * 2
    var = 1 / snr
    std = np.sqrt(var)
    receive_signal_noised = receive_signal + std * noise_com    # (K, 2rx)

    # hty(K, 2tx), hth(K, 2tx, 2tx)
    # hty = np.matmul(np.transpose(h_com, [2, 1, 0]), np.expand_dims(receive_signal_noised, axis=-1)).squeeze(-1)
    # hth = np.matmul(np.transpose(h_com, [2, 1, 0]), np.transpose(h_com, [2, 0, 1]))

    return receive_signal_noised, np.transpose(h_com, [2, 0, 1]), data_real, data_imag


def get_mmse(tx=8, rx=8, K=1000, rate=1, EbN0=15):
    """
    Using the augmented receive signal and channel matrix to generate the data
    """
    l = np.power(2, rate)    # PAM alphabet size
    # gray_table = gray_map(rate)

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])    # (2tx, K)

    h_real = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_imag = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_com = np.vstack([np.hstack([h_real, -1*h_imag]),  # (2rx, 2tx, K)
                      np.hstack([h_imag, h_real])])

    noise_real = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_imag = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_com = np.hstack([noise_real, noise_imag])     # (K, 2rx)
    # y(K, 2rx)
    receive_signal = np.matmul(np.transpose(h_com, [2, 0, 1]), np.expand_dims(data_com.T, axis=-1)).squeeze(-1)

    snr = np.power(10, EbN0/10) * 3 / 2 / (l*l - 1) / tx * rate * 2
    var = 1 / snr
    std = np.sqrt(var)
    receive_signal_noised = receive_signal + std * noise_com    # (K, 2rx)

    # --------------------------------------------------- MMSE --------------------------------------------------------
    extended_part_real = np.sqrt(var*3/2/(l**2-1))*np.eye(tx)

    mmse_real = np.vstack([h_real, np.tile(np.expand_dims(extended_part_real, axis=-1), [1, 1, K])])    # (rx+tx, tx, K)
    mmse_imag = np.vstack([h_imag, np.zeros(([tx, tx, K]))])    # (rx+tx, tx, K)
    h_mmse = np.vstack([np.hstack([mmse_real, -1*mmse_imag]),
                        np.hstack([mmse_imag, mmse_real])])     # (2rx+2tx, 2tx, K)
    receive_zeros = np.zeros([K, tx])
    receive_mmse = np.hstack([receive_signal_noised[:, :rx], receive_zeros,
                              receive_signal_noised[:, rx:], receive_zeros])    # (K, 2rx+2tx)

    return receive_mmse, np.transpose(h_mmse, [2, 0, 1]), data_real, data_imag


def get_wide_mmse(tx=8, rx=8, K=1000, rate=1):
    """
    Using the augmented receive signal and channel matrix to generate the data.
    The Eb/N0 lies between [-1, 20] dB
    """
    l = np.power(2, rate)    # PAM alphabet size
    # gray_table = gray_map(rate)
    EbN0 = np.random.uniform(-1, 20, [K, 1])

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])    # (2tx, K)

    h_real = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_imag = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_com = np.vstack([np.hstack([h_real, -1*h_imag]),  # (2rx, 2tx, K)
                      np.hstack([h_imag, h_real])])

    noise_real = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_imag = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_com = np.hstack([noise_real, noise_imag])     # (K, 2rx)
    # y(K, 2rx)
    receive_signal = np.matmul(np.transpose(h_com, [2, 0, 1]), np.expand_dims(data_com.T, axis=-1)).squeeze(-1)

    snr = np.power(10, EbN0/10) * 3 / 2 / (l*l - 1) / tx * rate * 2
    var = 1 / snr
    std = np.sqrt(var)
    receive_signal_noised = receive_signal + std * noise_com    # (K, 2rx)

    # --------------------------------------------------- MMSE --------------------------------------------------------
    extended_part_real = np.einsum('ij,kl->ijk', np.eye(tx), np.sqrt(var*3/2/(l**2-1)))

    mmse_real = np.vstack([h_real, extended_part_real])
    # mmse_real = np.vstack([h_real, np.tile(np.expand_dims(extended_part_real, axis=-1), [1, 1, K])])    # (rx+tx, tx, K)
    mmse_imag = np.vstack([h_imag, np.zeros(([tx, tx, K]))])    # (rx+tx, tx, K)
    h_mmse = np.vstack([np.hstack([mmse_real, -1*mmse_imag]),
                        np.hstack([mmse_imag, mmse_real])])     # (2rx+2tx, 2tx, K)
    receive_zeros = np.zeros([K, tx])
    receive_mmse = np.hstack([receive_signal_noised[:, :rx], receive_zeros,
                              receive_signal_noised[:, rx:], receive_zeros])    # (K, 2rx+2tx)

    return receive_mmse, np.transpose(h_mmse, [2, 0, 1]), data_real, data_imag


def noise_free_data(tx=8, rx=8, K=1000, rate=1):
    """
    Receive Signal Without Noise
    :param tx:transmitting antennas
    :param rx:receiving antennas
    :param K:data point
    :param rate:modulation rate
    :return:y = Hx
    """
    l = np.power(2, rate)    # PAM alphabet size
    # gray_table = gray_map(rate)

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])    # (2tx, K)

    h_real = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_imag = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_com = np.vstack([np.hstack([h_real, -1*h_imag]),  # (2rx, 2tx, K)
                      np.hstack([h_imag, h_real])])

    # y(K, 2rx)
    receive_signal = np.matmul(np.transpose(h_com, [2, 0, 1]), np.expand_dims(data_com.T, axis=-1)).squeeze(-1)  # (K, 2rx)

    return receive_signal, np.transpose(h_com, [2, 0, 1]), data_real, data_imag


def get_data_channel(tx=8, rx=8, K=20000, rate=1, EbN0=15):
    """
    Output: y, HtH, Hty, H
    """
    l = np.power(2, rate)    # PAM alphabet size

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])

    h_real = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_imag = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_com = np.vstack([np.hstack([h_real, -1*h_imag]),  # (2rx, 2tx, K)
                      np.hstack([h_imag, h_real])])

    noise_real = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_imag = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_com = np.hstack([noise_real, noise_imag])     # (K, 2rx)
    # y(K, 2rx)
    receive_signal = np.matmul(np.transpose(h_com, [2, 0, 1]), np.expand_dims(data_com.T, axis=-1)).squeeze(-1)

    snr = np.power(10, EbN0/10) * 3 / 2 / (l*l - 1) / tx * rate * 2
    var = 1 / snr
    std = np.sqrt(var)
    receive_signal_noised = receive_signal + std * noise_com    # (K, 2rx)

    # hty(K, 2tx), hth(K, 2tx, 2tx)
    hty = np.matmul(np.transpose(h_com, [2, 1, 0]), np.expand_dims(receive_signal_noised, axis=-1)).squeeze(-1)
    hth = np.matmul(np.transpose(h_com, [2, 1, 0]), np.transpose(h_com, [2, 0, 1]))

    Es = 2 * (l ** 2 - 1) / 3
    A = hth + var / Es * np.eye(2 * tx)

    return receive_signal_noised, A, hty, np.transpose(h_com, [2, 0, 1]), data_real, data_imag


def get_data_channel_noisefree(tx=8, rx=8, K=20000, rate=1):
    """
    Noise free data
    Output: y, HtH, Hty, H
    """
    l = np.power(2, rate)    # PAM alphabet size

    data_real = np.random.randint(0, l, [tx, K])
    data_imag = np.random.randint(0, l, [tx, K])
    data_com = np.vstack([2 * data_real - l + 1, 2 * data_imag - l + 1])

    h_real = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_imag = np.sqrt(1/2) * np.random.randn(rx, tx, K)
    h_com = np.vstack([np.hstack([h_real, -1*h_imag]),  # (2rx, 2tx, K)
                      np.hstack([h_imag, h_real])])

    noise_real = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_imag = np.sqrt(1/2) * np.random.randn(K, rx)
    noise_com = np.hstack([noise_real, noise_imag])     # (K, 2rx)
    # y(K, 2rx)
    receive_signal = np.matmul(np.transpose(h_com, [2, 0, 1]), np.expand_dims(data_com.T, axis=-1)).squeeze(-1)

    # hty(K, 2tx), hth(K, 2tx, 2tx)
    hty = np.matmul(np.transpose(h_com, [2, 1, 0]), np.expand_dims(receive_signal, axis=-1)).squeeze(-1)
    hth = np.matmul(np.transpose(h_com, [2, 1, 0]), np.transpose(h_com, [2, 0, 1]))

    A = hth

    return receive_signal, A, hty, np.transpose(h_com, [2, 0, 1]), data_real, data_imag
