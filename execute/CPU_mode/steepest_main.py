from functions.data_preparation import get_data_channel, get_data_channel_noisefree
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
class DetDataset(Dataset):
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


if __name__ == '__main__':
    TX = 16
    RX = 16
    N_TRAIN = 150000
    N_TEST = 5000
    TRAIN_SPLIT = 0.9
    RATE = 1
    EBN0_TRAIN = 10
    LENGTH = 2 ** RATE
    BATCH_SIZE = 20
    EPOCHS = 100
    PROJECT_TIMES = 4
    RNN_HIDDEN_SIZE = 8 * TX
    STEP_SIZE = 0.012
    ITERATIONS = 7
    SPARSITY = 0.4

    # train_y, train_A, train_b, train_h, train_Data_real, train_Data_imag = get_data_channel(tx=TX,
    #                                                                                         rx=RX,
    #                                                                                         K=N_TRAIN,
    #                                                                                         rate=RATE,
    #                                                                                         EbN0=EBN0_TRAIN)
    train_y, train_A, train_b, train_h, train_Data_real, train_Data_imag = get_data_channel_noisefree(tx=TX,
                                                                                            rx=RX,
                                                                                            K=N_TRAIN,
                                                                                            rate=RATE)
    test_y, test_A, test_b, test_h, test_Data_real, test_Data_imag = get_data_channel(tx=TX,
                                                                                      rx=RX,
                                                                                      K=N_TEST,
                                                                                      rate=RATE,
                                                                                      EbN0=EBN0_TRAIN)

    train_set = DetDataset(train_y, train_A, train_b, train_h, train_Data_real, train_Data_imag)
    test_set = DetDataset(test_y, test_A, test_b, test_h, test_Data_real, test_Data_imag)

    trainSet, valSet = Data.random_split(train_set, [int(N_TRAIN * TRAIN_SPLIT), round(N_TRAIN * (1 - TRAIN_SPLIT))])
    train_loader = Data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = Data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    # ------------------------------------- Establish Network ----------------------------------------------
    # PATH = '../../SAD_pretrained/SteepestDescent/ber_model/tx%i/rx%i/rate%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f/sparse%.3f' % (TX, RX, RATE,
    #                                                                                                         ITERATIONS,
    #                                                                                                         PROJECT_TIMES,
    #                                                                                                         BATCH_SIZE,
    #                                                                                                         RNN_HIDDEN_SIZE,
    #                                                                                                         STEP_SIZE,
    #                                                                                                         SPARSITY)
    # prenet = ProjectionXI.DetModel(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES, SPARSITY)

    model = my_models.SteepestDescentDetector(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES, SPARSITY)
    # model.load_state_dict(torch.load(PATH + str('/model.pt')))

    # model = ProjectionXI.SteepestDescent(TX)
    model_pre = model

    optim_det = torch.optim.Adam(model.parameters(), lr=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim_det, step_size=10, gamma=0.2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_det, gamma=0.97)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_det, [10, 20, 35, 50, 70, 90], 0.1)

    ber_pre = 1.

    # ------------------------------------- Train ----------------------------------------------------------
    print('Begin Training:')
    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            A, b, label = data['A'], data['b'], data['label']
            # zero the parameter gradients
            model.zero_grad()

            # forward + backward + optimize
            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
            h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

            x, h, outputs = model(A, b, x_ini, h_ini, ITERATIONS)

            # x, h, outputs = model(inputs, x_ini, h_ini, ITERATIONS)
            loss = loss_cal.weighted_mse(outputs, label, RATE)
            loss.backward()
            optim_det.step()

            # print statistics
            running_loss += loss.item()
            if i % (round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)) == round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE) - 1:
                # writer.add_scalar('loss/train_loss', running_loss / round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE), epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)))
                running_loss = 0.0
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        val_prediction = []
        val_data_real = []
        val_data_imag = []
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                model.eval()
                A, b, label, data_real, data_imag = data['A'], data['b'], data['label'], data['data_real'], data['data_imag']

                x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
                x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
                h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

                x, h, outputs = model(A, b, x_ini, h_ini, ITERATIONS)
                loss = loss_cal.weighted_mse(outputs, label, RATE)

                val_prediction += [x]

                val_data_real += [data_real]
                val_data_imag += [data_imag]

                val_loss += loss.numpy()
                val_steps += 1
        # writer.add_scalar('loss/val_loss', val_loss/val_steps, epoch)
        print('validation loss: %.3f' % (val_loss / val_steps))

        scheduler.step()

        val_data_real = torch.cat(val_data_real).T.numpy()
        val_data_imag = torch.cat(val_data_imag).T.numpy()
        val_prediction = torch.cat(val_prediction).numpy()
        val_prediction = ((val_prediction + LENGTH - 1) / 2).round()
        ber = gray_ber(val_prediction, val_data_real, val_data_imag, rate=RATE)
        if ber < ber_pre:
            ber_pre = ber
            model_pre = model
            print('model updated')
        else:
            print('model NOT updated!')

    model = model_pre
    print('Training finished')

    # --------------------------------------------------- Test ---------------------------------------------------------
    sd = my_models.SteepestDescent(TX)
    with torch.no_grad():
        model.eval()

        projection_loss = 0.0
        sd_loss = 0.0
        projection_ml_loss = 0.0
        sd_ml_loss = 0.0

        test_steps = 0

        predictions = []
        predictions_sd = []
        for i, data in enumerate(test_loader, 0):
            A, b, label, data_real, data_imag, y, h_com = data['A'], data['b'], data['label'], data['data_real'], data['data_imag'], data['y'], data['h']
            inputs = (y, h_com)

            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
            h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

            x, h, outputs = model(A, b, x_ini, h_ini, ITERATIONS)
            loss = loss_cal.weighted_mse(outputs, label, RATE)

            results = outputs[3]

            loss_mse_p = loss_cal.common_loss(results, label, RATE)
            loss_ml_p = loss_cal.ml_loss_single(results, y, h_com)
            x_sd = sd(A, b, results, 6)
            loss_mse_g = loss_cal.common_loss(x_sd, label, RATE)
            loss_ml_g = loss_cal.ml_loss_single(x_sd, y, h_com)

            predictions += [results]
            predictions_sd += [x_sd]

            projection_loss += loss_mse_p.numpy()
            sd_loss += loss_mse_g.numpy()
            projection_ml_loss += loss_ml_p.numpy()
            sd_ml_loss += loss_ml_g.numpy()

            test_steps += 1
        print('projection mse loss: %.5f' % (projection_loss / test_steps))
        print('steepest descent mse loss: %.5f' % (sd_loss / test_steps))
        print('projection ML loss: %.5f' % (projection_ml_loss / test_steps))
        print('steepest descent ML loss: %.5f' % (sd_ml_loss / test_steps))

        predictions = torch.cat(predictions).cpu().numpy()
        predictions = ((predictions + LENGTH - 1)/2).round()
        ber = gray_ber(predictions, test_Data_real, test_Data_imag, rate=RATE)

        predictions_sd = torch.cat(predictions_sd).cpu().numpy()
        predictions_sd = ((predictions_sd + LENGTH - 1) / 2).round()
        ber_cg = gray_ber(predictions_sd, test_Data_real, test_Data_imag, rate=RATE)

    # ------------------------------------------------ Whole Test ------------------------------------------------------
    with torch.no_grad():
        model.eval()
        TEST_EBN0 = np.linspace(0, 18, 19)
        BER = []
        BER_SD = []
        for ebn0 in TEST_EBN0:
            test_y, test_A, test_b, test_h, test_Data_real, test_Data_imag = get_data_channel(tx=TX,
                                                                                              rx=RX,
                                                                                              K=N_TEST,
                                                                                              rate=RATE,
                                                                                              EbN0=ebn0)
            test_set = DetDataset(test_y, test_A, test_b, test_h, test_Data_real, test_Data_imag)
            test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

            with torch.no_grad():
                model.eval()
                test_loss = 0.0
                test_steps = 0
                prediction = []
                prediction_sd = []
                for i, data in enumerate(test_loader, 0):
                    A, b, label = data['A'], data['b'], data['label']
                    x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
                    x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
                    h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

                    x, h, outputs = model(A, b, x_ini, h_ini, ITERATIONS)
                    loss = loss_cal.weighted_mse(outputs, label, RATE)
                    x_sd = sd(A, b, x, ITERATIONS)
                    prediction += [x]
                    prediction_sd += [x_sd]
                    test_loss += loss.numpy()
                    test_steps += 1
                print('test loss: %.3f' % (test_loss / test_steps))

                prediction = torch.cat(prediction).cpu().numpy()
                prediction = ((prediction + LENGTH - 1) / 2).round()
                ber = gray_ber(prediction, test_Data_real, test_Data_imag, rate=RATE)
                BER += [ber]

                prediction_sd = torch.cat(prediction_sd).cpu().numpy()
                prediction_sd = ((prediction_sd + LENGTH - 1) / 2).round()
                ber_sd = gray_ber(prediction_sd, test_Data_real, test_Data_imag, rate=RATE)
                BER_SD += [ber_sd]

    # --------------------------------------- Save Model & Data --------------------------------------------------------
    PATH = '../../SAD_pretrained/SteepestDescent/ber_model/noise_free/tx%i/rx%i/rate%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f/sparse%.3f' % (TX, RX, RATE,
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
    torch.save(model.state_dict(), PATH + str('/model2.pt'))
    # use the following line to load model
    # detnet.load_state_dict(torch.load(PATH + str('/model.pt')))
    scio.savemat(PATH+str('/BER2.mat'), {'ber': BER})
    scio.savemat(PATH+str('/BER_SD2.mat'), {'ber_sd': BER_SD})


