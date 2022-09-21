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
    ITERATIONS = 3
    SPARSITY = 0.4

    train_y, train_h_com, train_Data_real, train_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TRAIN, rate=RATE, EbN0=EBN0_TRAIN)
    # train_y, train_h_com, train_Data_real, train_Data_imag = data_preparation.noise_free_data(tx=TX, rx=RX, K=N_TRAIN, rate=RATE)

    test_y, test_h_com, test_Data_real, test_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TEST, rate=RATE, EbN0=EBN0_TRAIN)

    train_set = DetDataset(train_y, train_h_com, train_Data_real, train_Data_imag)
    test_set = DetDataset(test_y, test_h_com, test_Data_real, test_Data_imag)

    trainSet, valSet = Data.random_split(train_set, [int(N_TRAIN * TRAIN_SPLIT), round(N_TRAIN * (1 - TRAIN_SPLIT))])
    train_loader = Data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = Data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    # ------------------------------------- Establish Network ----------------------------------------------
    # PATH = '../../SAD_pretrained/Newton/ber_model/tx%i/rx%i/rate%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f/sparse%.3f' % (TX, RX, RATE,
    #                                                                                                         ITERATIONS,
    #                                                                                                         PROJECT_TIMES,
    #                                                                                                         BATCH_SIZE,
    #                                                                                                         RNN_HIDDEN_SIZE,
    #                                                                                                         STEP_SIZE,
    #                                                                                                         SPARSITY)
    # prenet = ProjectionXI.DetModel(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES, SPARSITY)

    model = my_models.QuasiNewtonProjectionDetector(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES, SPARSITY)
    # model.load_state_dict(torch.load(PATH + str('/model.pt')))

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
            y, h_com, label = data['y'], data['h_com'], data['label']
            inputs = (y, h_com)
            # zero the parameter gradients
            model.zero_grad()

            # forward + backward + optimize
            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
            h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

            x, h, outputs, _ = model(inputs, x_ini, h_ini, ITERATIONS)
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
                y, h_com, label, data_real, data_imag = data['y'], data['h_com'], data['label'], data['data_real'], data['data_imag']
                inputs = (y, h_com)

                x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
                x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
                h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

                x, h, outputs, _ = model(inputs, x_ini, h_ini, ITERATIONS)
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
    # gd = ProjectionXI.GDDetector()
    bfgs = my_models.QuasiNewton(TX)
    with torch.no_grad():
        model.eval()

        projection_loss = 0.0
        gd_loss = 0.0
        projection_ml_loss = 0.0
        gd_ml_loss = 0.0

        test_steps = 0

        predictions = []
        predictions_bfgs = []
        for i, data in enumerate(test_loader, 0):
            y, h_com, label, data_real, data_imag = data['y'], data['h_com'], data['label'], data['data_real'], data['data_imag']
            inputs = (y, h_com)

            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
            h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

            x, h, outputs, temp = model(inputs, x_ini, h_ini, ITERATIONS)
            loss = loss_cal.weighted_mse(outputs, label, RATE)

            results = outputs[3]

            loss_mse_p = loss_cal.common_loss(results, label, RATE)
            loss_ml_p = loss_cal.ml_loss_single(results, y, h_com)

            (alpha, h_k) = temp
            x_bfgs = bfgs(inputs, ITERATIONS, results, alpha, h_k)
            # x_gd = gd(inputs, results, STEP_SIZE, 10)
            loss_mse_g = loss_cal.common_loss(x_bfgs, label, RATE)
            loss_ml_g = loss_cal.ml_loss_single(x_bfgs, y, h_com)

            predictions += [results]
            predictions_bfgs += [x_bfgs]

            projection_loss += loss_mse_p.numpy()
            gd_loss += loss_mse_g.numpy()
            projection_ml_loss += loss_ml_p.numpy()
            gd_ml_loss += loss_ml_g.numpy()

            test_steps += 1
        print('projection mse loss: %.5f' % (projection_loss / test_steps))
        print('gd mse loss: %.5f' % (gd_loss / test_steps))
        print('projection ML loss: %.5f' % (projection_ml_loss / test_steps))
        print('gd ML loss: %.5f' % (gd_ml_loss / test_steps))

        predictions = torch.cat(predictions).cpu().numpy()
        predictions = ((predictions + LENGTH - 1)/2).round()
        ber = gray_ber(predictions, test_Data_real, test_Data_imag, rate=RATE)

        predictions_bfgs = torch.cat(predictions_bfgs).cpu().numpy()
        predictions_bfgs = ((predictions_bfgs + LENGTH - 1) / 2).round()
        ber_bfgs = gray_ber(predictions_bfgs, test_Data_real, test_Data_imag, rate=RATE)

    # ------------------------------------------------ Whole Test ------------------------------------------------------
    with torch.no_grad():
        model.eval()
        TEST_EBN0 = np.linspace(0, 18, 19)
        BER = []
        BER_BFGS = []
        for ebn0 in TEST_EBN0:
            # test_y, test_h_com, test_Data_real, test_Data_imag = data_preparation.get_data(tx=TX, rx=RX,
            #                                                                                       K=N_TEST, rate=RATE,
            #                                                                                       EbN0=ebn0)
            test_y, test_h_com, test_Data_real, test_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TEST,
                                                                                           rate=RATE, EbN0=ebn0)

            test_set = DetDataset(test_y, test_h_com, test_Data_real, test_Data_imag)
            test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
            with torch.no_grad():
                model.eval()
                test_loss = 0.0
                test_steps = 0
                prediction = []
                prediction_bfgs = []
                for i, data in enumerate(test_loader, 0):
                    y, h_com, label = data['y'], data['h_com'], data['label']
                    inputs = (y, h_com)
                    x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX])  # 16QAM
                    x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
                    h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

                    x, h, outputs, temp = model(inputs, x_ini, h_ini, ITERATIONS)
                    loss = loss_cal.weighted_mse(outputs, label, RATE)

                    (alpha, h_k) = temp
                    x_bfgs = bfgs(inputs, ITERATIONS, x, alpha, h_k)
                    # x_gd = gd(inputs, x, STEP_SIZE, 10)
                    prediction += [x]
                    prediction_bfgs += [x_bfgs]
                    test_loss += loss.numpy()
                    test_steps += 1
                print('test loss: %.3f' % (test_loss / test_steps))

                prediction = torch.cat(prediction).cpu().numpy()
                prediction = ((prediction + LENGTH - 1) / 2).round()
                ber = gray_ber(prediction, test_Data_real, test_Data_imag, rate=RATE)
                BER += [ber]

                prediction_bfgs = torch.cat(prediction_bfgs).cpu().numpy()
                prediction_bfgs= ((prediction_bfgs + LENGTH - 1) / 2).round()
                ber_bfgs= gray_ber(prediction_bfgs, test_Data_real, test_Data_imag, rate=RATE)
                BER_BFGS += [ber_bfgs]

    # --------------------------------------- Save Model & Data --------------------------------------------------------
    PATH = '../../SAD_pretrained/Newton/ber_model/tx%i/rx%i/rate%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f/sparse%.3f' % (TX, RX, RATE,
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
    scio.savemat(PATH+str('/BER_mmse.mat'), {'ber': BER})
    scio.savemat(PATH+str('/BER_bfgs_mmse.mat'), {'ber_gd': BER_BFGS})
