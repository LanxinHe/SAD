from functions import data_preparation
from functions.test_functions import gray_ber
from functions import loss_cal
from model import my_models
# from model.CPU_model import RNN_different
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


# --------------------------------------------- Dataset ------------------------------------------------------
class DetDataset(Dataset):
    def __init__(self, y, h_com, data_real, data_imag, transform=None):
        self.y = y
        self.h_com = h_com
        self.label = torch.cat([torch.from_numpy(data_real.T),
                                torch.from_numpy(data_imag.T)],
                               dim=1).long()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'y': torch.from_numpy(self.y[idx, :]).to(torch.float32),
            'h_com': torch.from_numpy(self.h_com[idx, :, :]).to(torch.float32),
            'label': self.label[idx, :],
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
    RNN_HIDDEN_SIZE = 12 * TX
    STEP_SIZE = 0.012
    ITERATIONS = 10
    SPARSITY = 0.3

    # train_y, train_h_com, train_Data_real, train_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TRAIN, rate=RATE, EbN0=EBN0_TRAIN)
    train_y, train_h_com, train_Data_real, train_Data_imag = data_preparation.noise_free_data(tx=TX, rx=RX, K=N_TRAIN, rate=RATE)

    test_y, test_h_com, test_Data_real, test_Data_imag = data_preparation.get_mmse(tx=TX, rx=RX, K=N_TEST, rate=RATE, EbN0=EBN0_TRAIN)

    train_set = DetDataset(train_y, train_h_com, train_Data_real, train_Data_imag)
    test_set = DetDataset(test_y, test_h_com, test_Data_real, test_Data_imag)

    trainSet, valSet = Data.random_split(train_set, [int(N_TRAIN * TRAIN_SPLIT), round(N_TRAIN * (1 - TRAIN_SPLIT))])
    train_loader = Data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = Data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    # ------------------------------------- Establish Network ----------------------------------------------
    # PATH = '../../PureSAD/tx%i/rx%i/rate%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f/sparse%.3f' % (TX, RX, RATE,
    #                                                                                                         ITERATIONS,
    #                                                                                                         PROJECT_TIMES,
    #                                                                                                         BATCH_SIZE,
    #                                                                                                         RNN_HIDDEN_SIZE,
    #                                                                                                         STEP_SIZE,
    #                                                                                                         SPARSITY)


    sad = my_models.DetModel(TX, RNN_HIDDEN_SIZE, PROJECT_TIMES, SPARSITY)
    # sad.load_state_dict(torch.load(PATH + str('/model2.pt')))

    optim_det = torch.optim.Adam(sad.parameters(), lr=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim_det, step_size=10, gamma=0.2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_det, gamma=0.97)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_det, [10, 20, 35, 50, 70, 90], 0.1)

    # ------------------------------------- Train ----------------------------------------------------------
    print('Begin Training:')
    for epoch in range(EPOCHS):
        running_loss = 0.0
        sad.train()
        for i, data in enumerate(train_loader, 0):
            y, h_com, label = data['y'], data['h_com'], data['label']
            inputs = (y, h_com)

            # zero the parameter gradients
            sad.zero_grad()

            # forward + backward + optimize
            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])  # 16QAM
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
            h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

            # x, h, _ = prenet(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
            x, h, outputs = sad(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
            loss = loss_cal.weighted_mse(outputs, label, RATE)
            loss.backward()
            optim_det.step()
            # for p in range(PROJECT_TIMES):
            #     getattr(detnet, 'r_cell%i' % p).linear_h.rezeroWeights()
            #     getattr(detnet, 'r_cell%i' % p).linear_x.rezeroWeights()
            sad.r_cell.linear_h.rezeroWeights()
            sad.r_cell.linear_x.rezeroWeights()

            # print statistics
            running_loss += loss.item()
            if i % (round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)) == round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE) - 1:
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)))
                running_loss = 0.0
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                sad.eval()
                # prenet1.eval()
                y, h_com, label = data['y'], data['h_com'], data['label']
                inputs = (y, h_com)

                x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])  # 16QAM
                x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
                h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

                x, h, outputs = sad(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)
                loss = loss_cal.weighted_mse(outputs, label, RATE)
                val_loss += loss.numpy()
                val_steps += 1
        print('validation loss: %.5f' % (val_loss / val_steps))
        scheduler.step()

    print('Training finished')

    # --------------------------------------------------- Test ---------------------------------------------------------
    gd = my_models.GDDetector()
    with torch.no_grad():
        sad.eval()

        projection_loss = 0.0
        gd_loss = 0.0
        projection_ml_loss = 0.0
        gd_ml_loss = 0.0

        test_steps = 0

        predictions = []
        predictions_gd = []
        predictions_more = []
        for i, data in enumerate(test_loader, 0):
            y, h_com, label = data['y'], data['h_com'], data['label']
            inputs = (y, h_com)

            x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])
            x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
            h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

            x, h, outputs = sad(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)

            results = outputs[2]

            x, h, _ = sad(inputs, x.unsqueeze(-1), h, STEP_SIZE, ITERATIONS)
            loss_mse_p = loss_cal.common_loss(results, label, RATE)
            loss_ml_p = loss_cal.ml_loss_single(results, y, h_com)
            x_gd = gd(inputs, results, STEP_SIZE, 10)
            loss_mse_g = loss_cal.common_loss(x_gd, label, RATE)
            loss_ml_g = loss_cal.ml_loss_single(x_gd, y, h_com)

            x_more, h, outputs_more = sad(inputs, x_gd.unsqueeze(-1), h, STEP_SIZE, 10)
            predictions += [results]
            predictions_gd += [x_gd]
            predictions_more += [outputs_more[0]]

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

        predictions_gd = torch.cat(predictions_gd).cpu().numpy()
        predictions_gd = ((predictions_gd + LENGTH - 1) / 2).round()
        ber_cg = gray_ber(predictions_gd, test_Data_real, test_Data_imag, rate=RATE)

        predictions_more = torch.cat(predictions_more).cpu().numpy()
        predictions_more = ((predictions_more + LENGTH - 1)/2).round()
        ber_more = gray_ber(predictions_more, test_Data_real, test_Data_imag, rate=RATE)

    # ------------------------------------------------ Whole Test ------------------------------------------------------
    with torch.no_grad():
        sad.eval()
        TEST_EBN0 = np.linspace(0, 15, 16)
        BER = []
        for ebn0 in TEST_EBN0:
            test_y, test_h_com, test_Data_real, test_Data_imag = data_preparation.get_data(tx=TX, rx=RX,
                                                                                                  K=N_TEST, rate=RATE,
                                                                                                  EbN0=ebn0)
            test_set = DetDataset(test_y, test_h_com, test_Data_real, test_Data_imag)
            test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
            with torch.no_grad():
                sad.eval()
                test_loss = 0.0
                test_steps = 0
                prediction = []
                for i, data in enumerate(test_loader, 0):
                    y, h_com, label = data['y'], data['h_com'], data['label']
                    inputs = (y, h_com)

                    x_ini = torch.randint(2 ** RATE, [BATCH_SIZE, 2 * TX, 1])  # 16QAM
                    x_ini = (2 * x_ini - 2 ** RATE + 1).to(torch.float32)
                    h_ini = torch.zeros([BATCH_SIZE, RNN_HIDDEN_SIZE])

                    x, h, outputs = sad(inputs, x_ini, h_ini, STEP_SIZE, ITERATIONS)

                    loss = loss_cal.weighted_mse(outputs, label, RATE)
                    prediction += [x]
                    test_loss += loss.numpy()
                    test_steps += 1
                print('test loss: %.3f' % (test_loss / test_steps))

                prediction = torch.cat(prediction).numpy()
                prediction = ((prediction + LENGTH - 1) / 2).round()
                ber = gray_ber(prediction, test_Data_real, test_Data_imag, rate=RATE)
                BER += [ber]
    # --------------------------------------- Save Model & Data --------------------------------------------------------
    PATH = '../../SAD_pretrained/ber_model/tx%i/rx%i/rate%i/iterations%i/project_times%i/batch_size%i/rnn_hidden_size%i/step_size%.5f/sparse%.3f' % (TX, RX, RATE,
                                                                                                            ITERATIONS,
                                                                                                            PROJECT_TIMES,
                                                                                                            BATCH_SIZE,
                                                                                                            RNN_HIDDEN_SIZE,
                                                                                                            STEP_SIZE,
                                                                                                            SPARSITY)

    os.makedirs(PATH)
    data_ber = pd.DataFrame(BER, columns=['BER'])
    data_ber.to_csv(PATH+str('/ber_w.csv'))
    torch.save(sad.state_dict(), PATH + str('/model.pt'))
    # use the following line to load model
    sad.load_state_dict(torch.load(PATH + str('/model.pt')))
