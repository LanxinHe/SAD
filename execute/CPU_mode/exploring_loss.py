from functions import data_preparation
from functions import loss_cal
from functions.test_functions import gray_ber
from model import my_models

import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
# from tensorboardX import SummaryWriter
import scipy.io as scio


# --------------------------------------------- Dataset ------------------------------------------------------
class DetDatasetA(Dataset):
    def __init__(self, y, A, b, h, data_real, data_imag, transform=None):
        self.y = y
        self.A = A
        self.b = b
        self.h = h
        self.label = torch.cat([torch.from_numpy(data_real.T),
                                torch.from_numpy(data_imag.T)],
                               dim=1).float()
        self.data_real = data_real
        self.data_imag = data_imag
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'y': torch.from_numpy(self.y[idx, :]).to(torch.float32),
            'A': torch.from_numpy(self.A[idx, :, :]).to(torch.float32),
            'b': torch.from_numpy(self.b[idx, :]).to(torch.float32),
            'h': torch.from_numpy(self.h[idx, :, :]).to(torch.float32),
            'label': self.label[idx, :],
            'data_real': self.data_real[:, idx],
            'data_imag': self.data_imag[:, idx],
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


class DetDataset(Dataset):
    def __init__(self, y, h_com, data_real, data_imag, transform=None):
        self.y = y
        self.h_com = h_com
        self.label = torch.cat([torch.from_numpy(data_real.T),
                                torch.from_numpy(data_imag.T)],
                               dim=1).long()
        self.transform = transform
        self.data_real = data_real
        self.data_imag = data_imag

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'y': torch.from_numpy(self.y[idx, :]).to(torch.float32),
            'h_com': torch.from_numpy(self.h_com[idx, :, :]).to(torch.float32),
            'label': self.label[idx, :],
            'data_real': self.data_real[:, idx],
            'data_imag': self.data_imag[:, idx],
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    TX = 16
    RX = 16
    N_TEST = 5000
    TRAIN_SPLIT = 0.9
    RATE = 1
    EBN0_TRAIN = 10
    LENGTH = 2 ** RATE
    BATCH_SIZE = 20
    EPOCHS = 100
    PROJECT_TIMES = 4
    RNN_HIDDEN_SIZE = 12 * TX
    STEP_SIZE = 0.012
    ITERATIONS = 40
    SPARSITY = 0.3
    test_y, test_h_com, test_Data_real, test_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TEST, rate=RATE, EbN0=EBN0_TRAIN)

    testA_y, testA_A, testA_b, testA_h, testA_Data_real, testA_Data_imag = data_preparation.get_data_channel(tx=TX,
                                                                                      rx=RX,
                                                                                      K=N_TEST,
                                                                                      rate=RATE,
                                                                                      EbN0=EBN0_TRAIN)
    test_set = DetDataset(test_y, test_h_com, test_Data_real, test_Data_imag)
    testA_set = DetDatasetA(testA_y, testA_A, testA_b, testA_h, testA_Data_real, testA_Data_imag)

    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    testA_loader = Data.DataLoader(testA_set, batch_size=BATCH_SIZE, shuffle=False)
    # ------------------------------------- Establish Network ----------------------------------------------
    # PATH = '../../PureSAD/tx%i/rx%i/rate%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f/sparse%.3f' % (TX, RX, RATE,
    #                                                                                                         ITERATIONS,
    #                                                                                                         PROJECT_TIMES,
    #                                                                                                         BATCH_SIZE,
    #                                                                                                         RNN_HIDDEN_SIZE,
    #                                                                                                         STEP_SIZE,
    #                                                                                                         SPARSITY)
    # prenet = ProjectionXI.DetModel(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES, SPARSITY)
    # prenet.load_state_dict(torch.load(PATH + str('/model2.pt')))

    model_sd = my_models.SteepestDescent(TX)
    model_bgsd = my_models.QuasiNewtonMethod(TX)
    model_gd = my_models.GDDetector()
    model_mrida = my_models.MRIDAModel()
    model_cg = my_models.CGDetector(TX)

    # --------------------------------------------------- Test ---------------------------------------------------------
    with torch.no_grad():

        sd_loss = 0.0
        newton_loss = 0.0
        gd_loss = 0.0
        mrida_loss = 0.0
        mmse_loss = 0.0
        cg_loss = 0.0

        sd_ml_loss = 0.0
        newton_ml_loss = 0.0
        gd_ml_loss = 0.0
        mrida_ml_loss = 0.0
        mmse_ml_loss = 0.0
        cg_ml_loss = 0.0

        test_steps = 0
        predictions_sd = []
        predictions_newton =[]
        predictions_gd = []
        predictions_mrida = []
        predictions_mmse = []
        predictions_cg = []

        for i, data in enumerate(testA_loader, 0):
            A, b, label, data_real, data_imag, y, h_com = data['A'], data['b'], data['label'], data['data_real'], data['data_imag'], data['y'], data['h']
            inputs = (y, h_com)

            x_mmse = torch.bmm(torch.inverse(A), b.unsqueeze(-1)).squeeze(-1)
            predictions_mmse += [x_mmse]

            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)

            x_sd = model_sd(A, b, x_ini, ITERATIONS)
            x_mrida = model_mrida(A, b, 4, 10)
            x_cg = model_cg(A, b, 20)

            loss_mse_g = loss_cal.common_loss(x_sd, label, RATE)
            loss_ml_g = loss_cal.ml_loss_single(x_sd, y, h_com)
            loss_mse_m = loss_cal.common_loss(x_mrida, label, RATE)
            loss_ml_m = loss_cal.ml_loss_single(x_mrida, y, h_com)
            loss_mse_mmse = loss_cal.common_loss(x_mmse, label, RATE)
            loss_ml_mmse = loss_cal.ml_loss_single(x_mmse, y, h_com)
            loss_mse_cg = loss_cal.common_loss(x_cg, label, RATE)
            loss_ml_cg = loss_cal.ml_loss_single(x_cg, y, h_com)

            predictions_sd += [x_sd]
            predictions_mrida += [x_mrida]
            predictions_cg += [x_cg]

            sd_loss += loss_mse_g.numpy()
            sd_ml_loss += loss_ml_g.numpy()
            mrida_loss += loss_mse_m.numpy()
            mrida_ml_loss += loss_ml_m.numpy()
            mmse_loss += loss_mse_mmse.numpy()
            mmse_ml_loss += loss_ml_mmse.numpy()
            cg_loss += loss_mse_cg.numpy()
            cg_ml_loss += loss_ml_cg.numpy()

            test_steps += 1
        print('steepest descent mse loss: %.5f' % (sd_loss / test_steps))
        print('steepest descent ML loss: %.5f \n' % (sd_ml_loss / test_steps))
        print('MRIDA mse loss: %.5f' % (mrida_loss / test_steps))
        print('MRIDA ML loss: %.5f \n' % (mrida_ml_loss / test_steps))
        print('MMSE mse loss: %.5f' % (mmse_loss / test_steps))
        print('MMSE ML loss: %.5f\n' % (mmse_ml_loss / test_steps))
        print('CG mse loss: %.5f' % (cg_loss / test_steps))
        print('CG ML loss: %.5f\n' % (cg_ml_loss / test_steps))

        for i, data in enumerate(test_loader, 0):
            y, h_com, label, data_real, data_imag = data['y'], data['h_com'], data['label'], data['data_real'], data['data_imag']
            inputs = (y, h_com)

            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)

            x_bgsd = model_bgsd(inputs, ITERATIONS)
            x_gd = model_gd(inputs, x_ini, STEP_SIZE, ITERATIONS)

            loss_mse_b = loss_cal.common_loss(x_bgsd, label, RATE)
            loss_ml_b = loss_cal.ml_loss_single(x_bgsd, y, h_com)
            loss_mse_g = loss_cal.common_loss(x_gd, label, RATE)
            loss_ml_g = loss_cal.ml_loss_single(x_gd, y, h_com)

            predictions_newton += [x_bgsd]
            predictions_gd += [x_gd]

            newton_loss += loss_mse_b.numpy()
            newton_ml_loss += loss_ml_b.numpy()
            gd_loss += loss_mse_g.numpy()
            gd_ml_loss += loss_ml_g.numpy()

            test_steps += 1
        print('Newton mse loss: %.5f' % (newton_loss / test_steps))
        print('Newton ML loss: %.5f\n' % (newton_ml_loss / test_steps))
        print('GD mse loss: %.5f' % (gd_loss / test_steps))
        print('GD ML loss: %.5f\n' % (gd_ml_loss / test_steps))

        predictions_sd = torch.cat(predictions_sd).cpu().numpy()
        predictions_sd = ((predictions_sd + LENGTH - 1) / 2).round()
        ber_sd = gray_ber(predictions_sd, testA_Data_real, testA_Data_imag, rate=RATE)

        predictions_newton = torch.cat(predictions_newton).cpu().numpy()
        predictions_newton = ((predictions_newton + LENGTH - 1) / 2).round()
        ber_newton = gray_ber(predictions_newton, test_Data_real, test_Data_imag, rate=RATE)

        predictions_gd = torch.cat(predictions_gd).cpu().numpy()
        predictions_gd = ((predictions_gd + LENGTH - 1) / 2).round()
        ber_gd = gray_ber(predictions_gd, test_Data_real, test_Data_imag, rate=RATE)

        predictions_mrida = torch.cat(predictions_mrida).cpu().numpy()
        predictions_mrida = ((predictions_mrida + LENGTH - 1) / 2).round()
        ber_mrida = gray_ber(predictions_mrida, testA_Data_real, testA_Data_imag, rate=RATE)

        predictions_mmse = torch.cat(predictions_mmse).cpu().numpy()
        predictions_mmse = ((predictions_mmse + LENGTH - 1) / 2).round()
        ber_mmse = gray_ber(predictions_mmse, testA_Data_real, testA_Data_imag, rate=RATE)

        predictions_cg = torch.cat(predictions_cg).cpu().numpy()
        predictions_cg = ((predictions_cg + LENGTH - 1) / 2).round()
        ber_cg = gray_ber(predictions_cg, testA_Data_real, testA_Data_imag, rate=RATE)

    # --------------------------------------- Save Model & Data --------------------------------------------------------
    PATH = '../../SAD_pretrained/SteepestDescent/ber_model/tx%i/rx%i/rate%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f/sparse%.3f' % (TX, RX, RATE,
                                                                                                            ITERATIONS,
                                                                                                            PROJECT_TIMES,
                                                                                                            BATCH_SIZE,
                                                                                                            RNN_HIDDEN_SIZE,
                                                                                                            STEP_SIZE,
                                                                                                            SPARSITY)
    os.makedirs(PATH)
    data_ber = pd.DataFrame(BER, columns=['BER'])
    data_ber.to_csv(PATH+str('/BER.csv'))
    scio.savemat(PATH + str('/ber.mat'), {'ber': ber})
    torch.save(model.state_dict(), PATH + str('/model.pt'))
    # use the following line to load model
    # detnet.load_state_dict(torch.load(PATH + str('/model.pt')))


# def calculate_lossA(loader, model, iter_method):
#     mse_loss = 0.0
#     ml_loss = 0.0
#
#     test_step = 0
#     predictions = []
#
#     for i, data in enumerate(loader, 0):
#         A, b, label, data_real, data_imag, y, h_com = data['A'], data['b'], data['label'], data['data_real'], data[
#             'data_imag'], data['y'], data['h']
#         inputs = (y, h_com)
#
#         x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
#         x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
#
#         x = model_sd(A, b, x_ini, ITERATIONS)
#
#
