import torch
import torch.nn as nn
import sparselinear

import abc
import math
import numpy as np


"""
Use weighted loss function and Sparsely Connected Network with the attribute of weight_sparsity and act_sparsity
Here we take weight_sparsity and act_sparsity the same.
"""


class RecurrentCell(nn.Module):
    def __init__(self, tx, hidden_size, weight_sparsity):
        super(RecurrentCell, self).__init__()
        weight_sparsity = weight_sparsity
        self.linear_h = nn.Linear(hidden_size+2*tx, hidden_size)
        normalizeSparseWeights(self.linear_h, weight_sparsity)
        self.linear_h = SparseWeights(self.linear_h, weight_sparsity)

        self.k_winner = sparselinear.ActivationSparsity(act_sparsity=weight_sparsity)
        self.linear_x = nn.Linear(hidden_size, 2*tx)
        normalizeSparseWeights(self.linear_x, weight_sparsity)
        self.linear_x = SparseWeights(self.linear_x, weight_sparsity)

    def forward(self, z, h_pre):
        """
        :param z: in shape of (batch_size, 2tx)
        :param h_pre: the previous hidden state in shape of (batch_size, hidden_size)
        :return:
        """
        x = z
        # h (batch_size, hidden_size)
        cat = torch.cat([x, h_pre], dim=-1)
        h = self.k_winner(self.linear_h(cat))
        x = self.linear_x(h)

        return x, h


# --------------------------------------------------- SAD ------------------------------------------------------------
class DetModel(nn.Module):
    def __init__(self, tx, rnn_hidden_size, project_times, weight_sparsity):
        super(DetModel, self).__init__()
        self.r_cell = RecurrentCell(tx, rnn_hidden_size, weight_sparsity)
        self.project_times = project_times

    def forward(self, inputs, x, h, step_size, iterations):
        """
        :param inputs: (y, H); y(batch_size, 2rx+2tx) H(batch_size, 2rx+2tx, 2tx)
        :return:
        """
        y, H = inputs
        batch_size = y.shape[0]

        Hty = torch.bmm(torch.transpose(H, -1, 1), y.view(batch_size, -1, 1))   # (batch_size, 2tx, 1)
        HtH = torch.bmm(torch.transpose(H, -1, 1), H)   # (batch_size, 2tx, 2tx)

        outputs = []
        for p in range(self.project_times):
            for i in range(iterations):
                x = gradient_descent(Hty, HtH, step_size, x)
            x, h = self.r_cell(x.squeeze(-1), h)
            outputs += [x]
            x = x.unsqueeze(-1)
        x = x.squeeze(-1)
        return x, h, outputs


class GDDetector(nn.Module):
    def __init__(self):
        super(GDDetector, self).__init__()

    def forward(self, inputs, x, step_size, iterations):
        """
        :param inputs: (y, H); y(batch_size, 2rx+2tx) H(batch_size, 2rx+2tx, 2tx)
        :return:
        """
        y, H = inputs
        batch_size = y.shape[0]

        Hty = torch.bmm(torch.transpose(H, -1, 1), y.view(batch_size, -1, 1))   # (batch_size, 2tx, 1)
        HtH = torch.bmm(torch.transpose(H, -1, 1), H)   # (batch_size, 2tx, 2tx)
        x = x.unsqueeze(-1)
        for i in range(iterations):
            x = gradient_descent(Hty, HtH, step_size, x)

        x = x.squeeze(-1)
        return x


def gradient_descent(hty, hth, step_size, x_pre):
    z = x_pre + 2 * step_size * (hty - torch.bmm(hth, x_pre))
    return z


# ----------------------------------------- Attention Mechanism --------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, tx, hidden_size, weight_sparsity):
        super(Encoder, self).__init__()
        self.weight_sparsity = weight_sparsity
        self.encoder = nn.Linear(hidden_size + 2 * tx, hidden_size)
        normalizeSparseWeights(self.encoder, weight_sparsity)
        self.encoder = SparseWeights(self.encoder, weight_sparsity)
        self.k_winner = sparselinear.ActivationSparsity(act_sparsity=weight_sparsity)

    def forward(self, z, h_pre):
        x = z
        cat = torch.cat([x, h_pre], dim=-1)
        h = self.k_winner(self.encoder(cat))
        return h


class Decoder(nn.Module):
    def __init__(self, tx, hidden_size, weight_sparsity):
        super(Decoder, self).__init__()
        self.weight_sparsity = weight_sparsity
        self.decoder = nn.Linear(hidden_size, 2*tx)
        normalizeSparseWeights(self.decoder, weight_sparsity)
        self.decoder = SparseWeights(self.decoder, weight_sparsity)

        self.linear_c = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, c_t, h_t):
        cat = torch.cat([c_t, h_t], dim=-1)
        h_tilde = torch.relu(self.linear_c(cat))
        x = self.decoder(h_tilde)

        return x


class RSAttention(nn.Module):
    """
    Using Attention mechanism; the score matrix in score function is dense.
    """
    def __init__(self, tx, rnn_hidden_size, project_times, weight_sparsity):
        super(RSAttention, self).__init__()
        self.encoder = Encoder(tx, rnn_hidden_size, weight_sparsity)
        self.decoder = Decoder(tx, rnn_hidden_size, weight_sparsity)
        self.project_times = project_times
        self.Wa = nn.Parameter(torch.randn(rnn_hidden_size, rnn_hidden_size))  # (hidden_size, hidden_size)

    def forward(self, inputs, x, h, step_size, iterations):
        y, H = inputs
        batch_size = y.shape[0]

        Hty = torch.bmm(torch.transpose(H, -1, 1), y.view(batch_size, -1, 1))   # (batch_size, 2tx, 1)
        HtH = torch.bmm(torch.transpose(H, -1, 1), H)   # (batch_size, 2tx, 2tx)

        H_hs = h.unsqueeze(-1)
        outputs = []
        for p in range(self.project_times):
            for i in range(iterations):
                x = gradient_descent(Hty, HtH, step_size, x)

            h_t = self.encoder(x.squeeze(-1), h)
            c_t = self.attention(h_t, H_hs)
            H_hs = torch.cat([H_hs, h_t.unsqueeze(-1)], dim=-1)     # The current h_t becomes source hidden vector
            x = self.decoder(c_t, h_t)
            outputs += [x]
            x = x.unsqueeze(-1)
        x = x.squeeze(-1)
        return x, h, outputs

    def attention(self, h_t, H):
        # H = (h_s, ... )
        # h_t -> (batch_size, 1, hidden_size)
        score = torch.bmm(h_t.unsqueeze(-1).permute(0, 2, 1), torch.einsum('ij, kjl-> kil', self.Wa, H))    # (batch_size, 1, window)
        score = score.permute(0, 2, 1)  # (batch_size, window, 1)
        a_t = torch.softmax(score, dim=1)
        ctx_t = torch.bmm(H, a_t).squeeze(-1)  # (batch_size, hidden_size, 1)
        return ctx_t


class RSAttentionI(nn.Module):
    """
    Using Attention mechanism;
    Score function: MLP
    the score matrix in score function is dense.
    """
    def __init__(self, tx, rnn_hidden_size, project_times, weight_sparsity):
        super(RSAttentionI, self).__init__()
        self.encoder = Encoder(tx, rnn_hidden_size, weight_sparsity)
        self.decoder = Decoder(tx, rnn_hidden_size, weight_sparsity)
        self.project_times = project_times
        self.Wa = nn.Parameter(torch.randn(rnn_hidden_size, rnn_hidden_size))  # (hidden_size, hidden_size)
        score_mlp_size1 = int(rnn_hidden_size / 2)
        score_mlp_size2 = int(rnn_hidden_size/ 4)
        self.score_linear1 = nn.Linear(2 * rnn_hidden_size, score_mlp_size1)
        self.score_linear2 = nn.Linear(score_mlp_size1, score_mlp_size2)
        self.score_linear3 = nn.Linear(score_mlp_size2, 1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, inputs, x, h, step_size, iterations):
        y, H = inputs
        batch_size = y.shape[0]

        Hty = torch.bmm(torch.transpose(H, -1, 1), y.view(batch_size, -1, 1))   # (batch_size, 2tx, 1)
        HtH = torch.bmm(torch.transpose(H, -1, 1), H)   # (batch_size, 2tx, 2tx)

        H_hs = h.unsqueeze(-1)
        outputs = []
        for p in range(self.project_times):
            for i in range(iterations):
                x = gradient_descent(Hty, HtH, step_size, x)

            h_t = self.encoder(x.squeeze(-1), h)
            c_t = self.attention(h_t, H_hs)
            H_hs = torch.cat([H_hs, h_t.unsqueeze(-1)], dim=-1)     # The current h_t becomes source hidden vector
            x = self.decoder(c_t, h_t)
            outputs += [x]
            x = x.unsqueeze(-1)
        x = x.squeeze(-1)
        return x, h, outputs

    def attention(self, h_t, H):
        # H = (h_s, ... )
        # h_t -> (batch_size, 1, hidden_size)
        D = H.shape[-1]
        scores = []
        for d in range(D):
            cat = torch.cat([h_t, H[:, :, d].squeeze(-1)], dim=-1)
            score = self.leaky_relu(self.score_linear1(cat))
            score = self.leaky_relu(self.score_linear2(score))
            score = self.leaky_relu(self.score_linear3(score))
            scores += [score.unsqueeze(-1)]

        scores = torch.cat(scores, dim=-1)
        # score = torch.bmm(h_t.unsqueeze(-1).permute(0, 2, 1), torch.einsum('ij, kjl-> kil', self.Wa, H))    # (batch_size, 1, window)
        score = scores.permute(0, 2, 1)  # (batch_size, window, 1)
        a_t = torch.softmax(score, dim=1)
        ctx_t = torch.bmm(H, a_t).squeeze(-1)  # (batch_size, hidden_size, 1)
        return ctx_t


# --------------------------------------- BFGS Method -----------------------------------------------------------------
class QuasiNewtonMethod(nn.Module):
    def __init__(self, tx):
        super(QuasiNewtonMethod, self).__init__()
        self.tx = tx

    def forward(self, inputs, iteration, ro=0.5, c=1e-4, epsilon=1e-5):
        """
        using 3-D variable
        """
        y, h = inputs
        batch_size = y.shape[0]
        h_k = torch.eye(2 * self.tx).unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.randint(2, [batch_size, 2 * self.tx, 1])  # QPSK
        x = (2 * x - 2 + 1).to(torch.float32)

        y = y.unsqueeze(-1)
        alpha = torch.ones(batch_size)

        def f_value(current_point):
            return torch.norm(y - torch.bmm(h, current_point), dim=1)/2

        def gradient_value(current_point):
            gradient = torch.bmm(h.permute(0, 2, 1), torch.bmm(h, current_point)) - torch.bmm(h.permute(0, 2, 1), y)
            return gradient

        for i in range(iteration):
            p = - torch.bmm(h_k, gradient_value(x))
            while any(torch.greater(f_value(x + torch.multiply(alpha.unsqueeze(-1).repeat(1, 2 * self.tx).unsqueeze(-1), p)).squeeze(-1),
                              (f_value(x) + c * torch.multiply(alpha.unsqueeze(-1), torch.bmm(gradient_value(x).permute(0, 2, 1), p).squeeze(-1))).squeeze(-1))):
                alpha = torch.where(
                    torch.greater(f_value(x + torch.multiply(alpha.unsqueeze(-1).repeat(1, 2 * self.tx).unsqueeze(-1), p)).squeeze(-1),
                                  (f_value(x) + c * torch.multiply(alpha.unsqueeze(-1), torch.bmm(gradient_value(x).permute(0, 2, 1), p).squeeze(-1))).squeeze(-1)),
                    ro * alpha, alpha
                    )

            x_new = x + torch.multiply(alpha.unsqueeze(-1).repeat(1, 2 * self.tx).unsqueeze(-1), p)
            s = x_new - x
            y_k = gradient_value(x_new) - gradient_value(x)
            auxSc = torch.bmm(s.permute(0, 2, 1), y_k)

            h_k = h_k + torch.multiply((auxSc + torch.bmm(y_k.permute(0, 2, 1), torch.bmm(h_k, y_k))),
                                  torch.bmm(s, s.permute(0, 2, 1)))/auxSc**2 -\
                  (torch.bmm(h_k, torch.bmm(y_k, s.permute(0, 2, 1))) + torch.bmm(s, torch.bmm(y_k.permute(0, 2, 1), h_k)))/auxSc
            x = x_new

        return x.squeeze(-1)


class QuasiNewton(nn.Module):
    def __init__(self, tx):
        """
        subclass in Model
        """
        super(QuasiNewton, self).__init__()
        self.tx = tx

    def forward(self, inputs, iteration, x, alpha, h_k, ro=0.5, c=1e-4, epsilon=1e-5):
        """
        using 3-D variable
        """
        y, h = inputs
        # batch_size = y.shape[0]

        y = y.unsqueeze(-1)
        x = x.unsqueeze(-1)

        def f_value(current_point):
            return torch.norm(y - torch.bmm(h, current_point), dim=1)/2

        def gradient_value(current_point):
            gradient = torch.bmm(h.permute(0, 2, 1), torch.bmm(h, current_point)) - torch.bmm(h.permute(0, 2, 1), y)
            return gradient

        for i in range(iteration):
            p = - torch.bmm(h_k, gradient_value(x))
            while any(torch.greater(f_value(x + torch.multiply(alpha.unsqueeze(-1).repeat(1, 2 * self.tx).unsqueeze(-1), p)).squeeze(-1),
                              (f_value(x) + c * torch.multiply(alpha.unsqueeze(-1), torch.bmm(gradient_value(x).permute(0, 2, 1), p).squeeze(-1))).squeeze(-1))):
                alpha = torch.where(
                    torch.greater(f_value(x + torch.multiply(alpha.unsqueeze(-1).repeat(1, 2 * self.tx).unsqueeze(-1), p)).squeeze(-1),
                                  (f_value(x) + c * torch.multiply(alpha.unsqueeze(-1), torch.bmm(gradient_value(x).permute(0, 2, 1), p).squeeze(-1))).squeeze(-1)),
                    ro * alpha, alpha
                    )

            x_new = x + torch.multiply(alpha.unsqueeze(-1).repeat(1, 2 * self.tx).unsqueeze(-1), p)
            s = x_new - x
            y_k = gradient_value(x_new) - gradient_value(x)
            auxSc = torch.bmm(s.permute(0, 2, 1), y_k)

            h_k = h_k + torch.multiply((auxSc + torch.bmm(y_k.permute(0, 2, 1), torch.bmm(h_k, y_k))),
                                  torch.bmm(s, s.permute(0, 2, 1)))/auxSc**2 -\
                  (torch.bmm(h_k, torch.bmm(y_k, s.permute(0, 2, 1))) + torch.bmm(s, torch.bmm(y_k.permute(0, 2, 1), h_k)))/auxSc
            x = x_new

        return x.squeeze(-1)


class QuasiNewtonProjectionDetector(nn.Module):
    def __init__(self, tx, rnn_hidden_size, project_times, weight_sparsity):
        super(QuasiNewtonProjectionDetector, self).__init__()
        self.r_cell = RecurrentCell(tx, rnn_hidden_size, weight_sparsity)
        self.project_times = project_times
        self.tx = tx
        self.newton_method = QuasiNewton(tx)

    def forward(self, inputs, x, h, iterations):
        y, H = inputs
        batch_size = y.shape[0]

        outputs = []
        alpha = torch.ones(batch_size)
        h_k = torch.eye(2 * self.tx).unsqueeze(0).repeat(batch_size, 1, 1)

        for p in range(self.project_times):
            x = self.newton_method(inputs, iterations, x, alpha, h_k)
            x, h = self.r_cell(x, h)
            outputs += [x]

        temp = (alpha, h_k)

        return x, h, outputs, temp


# --------------------------------------------- Steepest Descent ------------------------------------------------------
class SteepestDescent(nn.Module):
    def __init__(self, tx):
        super(SteepestDescent, self).__init__()
        self.tx = tx

    def forward(self, A, b, x, iteration):
        b = b.unsqueeze(-1)
        x = x.unsqueeze(-1)
        for i in range(iteration):
            gradient_f = torch.bmm(A, x) - b
            temp1 = torch.bmm(gradient_f.permute(0, 2, 1), gradient_f)
            temp2 = torch.bmm(gradient_f.permute(0, 2, 1), torch.bmm(A, gradient_f))
            step_length = torch.div(temp1, temp2)
            x = x - torch.multiply(step_length, gradient_f)
        x = x.squeeze(-1)
        return x


class SteepestDescentDetector(nn.Module):
    def __init__(self, tx, rnn_hidden_size, project_times, weight_sparsity):
        super(SteepestDescentDetector, self).__init__()
        self.r_cell = RecurrentCell(tx, rnn_hidden_size, weight_sparsity)
        self.project_times = project_times
        self.tx = tx
        self.steepest_descent = SteepestDescent(tx)

    def forward(self, A, b, x, h, iterations):
        outputs = []
        for p in range(self.project_times):
            x = self.steepest_descent(A, b, x, iterations)
            x, h = self.r_cell(x, h)
            outputs += [x]

        return x, h, outputs


# --------------------------------------------- Conjugate Descent -----------------------------------------------------
class CGDetector(nn.Module):
    def __init__(self, tx):
        super(CGDetector, self).__init__()
        self.tx = tx

    def forward(self, A, b, iteration):
        """
        :param A: HtH+ sigma^2 , (batch_size, 2tx, 2tx)
        :param b: Hty, (batch_size, 2tx)
        :return:
        """
        batch_size = b.shape[0]
        s = torch.zeros(batch_size, 2*self.tx)
        r = torch.bmm(A, s.unsqueeze(-1)).squeeze(-1) - b    # Residual
        d = -r   # Direction

        for i in range(iteration):
            alpha = torch.divide(torch.bmm(r.unsqueeze(1), r.unsqueeze(-1)),
                                 torch.bmm(d.unsqueeze(1), torch.bmm(A, d.unsqueeze(-1))))  # alpha (b, 1, 1)
            s = s + torch.multiply(alpha.squeeze(-1), d)    #
            r_new = r + torch.multiply(alpha.squeeze(-1), torch.bmm(A, d.unsqueeze(-1)).squeeze(-1))    # (b, 2tx)
            r_new = torch.where(torch.greater(torch.abs(r_new), 1e-10), r_new, r)
            beta = torch.divide(torch.bmm(r_new.unsqueeze(1), r_new.unsqueeze(-1)),
                                torch.bmm(r.unsqueeze(1), r.unsqueeze(-1)))     # (b, 1, 1)
            d = -r_new + torch.multiply(beta.squeeze(-1), d)
            r = r_new

        return s


class CGMethod(nn.Module):
    def __init__(self, tx):
        super(CGMethod, self).__init__()
        self.tx = tx

    def forward(self, A, b, pre, iteration):
        """
        :param A: HtH+ sigma^2 , (batch_size, 2tx, 2tx)
        :param b: Hty, (batch_size, 2tx)
        :return:
        """
        (s, r, d) = pre

        for i in range(iteration):
            alpha = torch.divide(torch.bmm(r.unsqueeze(1), r.unsqueeze(-1)),
                                 torch.bmm(d.unsqueeze(1), torch.bmm(A, d.unsqueeze(-1))))  # alpha (b, 1, 1)
            s = s + torch.multiply(alpha.squeeze(-1), d)    #
            r_new = r + torch.multiply(alpha.squeeze(-1), torch.bmm(A, d.unsqueeze(-1)).squeeze(-1))    # (b, 2tx)
            # r_new = torch.where(torch.greater(torch.abs(r_new), 1e-10), r_new, r)
            beta = torch.divide(torch.bmm(r_new.unsqueeze(1), r_new.unsqueeze(-1)),
                                torch.bmm(r.unsqueeze(1), r.unsqueeze(-1)))     # (b, 1, 1)
            d = -r_new + torch.multiply(beta.squeeze(-1), d)
            r = r_new

        return s


class CGDetectorL(nn.Module):
    def __init__(self, tx, rnn_hidden_size, project_times, weight_sparsity):
        super(CGDetectorL, self).__init__()
        self.tx = tx
        self.project_times = project_times
        self.r_cell = RecurrentCell(tx, rnn_hidden_size, weight_sparsity)
        self.K = rnn_hidden_size

    def forward(self, A, b, iteration):
        """
        :param A: HtH+ sigma^2 , (batch_size, 2tx, 2tx)
        :param b: Hty, (batch_size, 2tx)
        :return:
        """
        batch_size = b.shape[0]
        s = torch.zeros(batch_size, 2*self.tx)
        r = torch.bmm(A, s.unsqueeze(-1)).squeeze(-1) - b    # Residual
        d = -r   # Direction

        h = torch.zeros(batch_size, self.K)

        predictions = []
        pre = []

        for p in range(self.project_times):
            for i in range(iteration):
                alpha = torch.divide(torch.bmm(r.unsqueeze(1), r.unsqueeze(-1)),
                                     torch.bmm(d.unsqueeze(1), torch.bmm(A, d.unsqueeze(-1))))  # alpha (b, 1, 1)
                s = s + torch.multiply(alpha.squeeze(-1), d)    #

                r_new = r + torch.multiply(alpha.squeeze(-1), torch.bmm(A, d.unsqueeze(-1)).squeeze(-1))    # (b, 2tx)
                r_new = torch.where(torch.greater(torch.abs(r_new), 1e-10), r_new, r)
                beta = torch.divide(torch.bmm(r_new.unsqueeze(1), r_new.unsqueeze(-1)),
                                    torch.bmm(r.unsqueeze(1), r.unsqueeze(-1)))     # (b, 1, 1)
                d = -r_new + torch.multiply(beta.squeeze(-1), d)
                r = r_new

            s, h = self.r_cell(s, h)

            # s = res_coef * s_tilde + (1 - res_coef) * s
            predictions += [s]
            pre += [(s, r, d)]

        return s, h, predictions, pre


class CGDetectorModified(nn.Module):
    def __init__(self, tx, rnn_hidden_size, project_times, weight_sparsity):
        super(CGDetectorModified, self).__init__()
        self.tx = tx
        self.project_times = project_times
        self.r_cell = RecurrentCell(tx, rnn_hidden_size, weight_sparsity)
        self.K = rnn_hidden_size

    def forward(self, A, b, iteration):
        """
        :param A: HtH+ sigma^2 , (batch_size, 2tx, 2tx)
        :param b: Hty, (batch_size, 2tx)
        :return:
        """
        batch_size = b.shape[0]
        s = torch.zeros(batch_size, 2 * self.tx)
        r = torch.bmm(A, s.unsqueeze(-1)).squeeze(-1) - b    # Residual
        d = -r   # Direction

        h = torch.zeros(batch_size, self.K)

        predictions = []
        pre = []

        for p in range(self.project_times):
            for i in range(iteration):
                alpha = torch.divide(torch.bmm(r.unsqueeze(1), r.unsqueeze(-1)),
                                     torch.bmm(d.unsqueeze(1), torch.bmm(A, d.unsqueeze(-1))))  # alpha (b, 1, 1)
                s = s + torch.multiply(alpha.squeeze(-1), d)  #

                r_new = r + torch.multiply(alpha.squeeze(-1), torch.bmm(A, d.unsqueeze(-1)).squeeze(-1))  # (b, 2tx)
                r_new = torch.where(torch.greater(torch.abs(r_new), 1e-10), r_new, r)
                beta = torch.divide(torch.bmm(r_new.unsqueeze(1), r_new.unsqueeze(-1)),
                                    torch.bmm(r.unsqueeze(1), r.unsqueeze(-1)))  # (b, 1, 1)
                d = -r_new + torch.multiply(beta.squeeze(-1), d)
                r = r_new

            s, h = self.r_cell(s, h)
            r = torch.bmm(A, s.unsqueeze(-1)).squeeze(-1) - b    # Residual
            d = -r
            # s = res_coef * s_tilde + (1 - res_coef) * s
            predictions += [s]
            pre += [(s, r, d)]

        return s, h, predictions, pre


# --------------------------------------------- Sparse Representation -------------------------------------------------
def rezeroWeights(m):
    """
    Function used to update the weights after each epoch.
    Call using :meth:`torch.nn.Module.apply` after each epoch if required
    For example: ``m.apply(rezeroWeights)``
    :param m: SparseWeightsBase module
    """
    if isinstance(m, SparseWeightsBase):
        if m.training:
          m.rezeroWeights()


def normalizeSparseWeights(m, weightSparsity):
    """
    Initialize the weights using kaiming_uniform initialization normalized to
    the number of non-zeros in the layer instead of the whole input size.
    Similar to torch.nn.Linear.reset_parameters() but applying weight sparsity
    to the input size
    """

    _, inputSize = m.weight.shape
    fan = int(inputSize * weightSparsity)
    gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
    std = gain / np.math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    nn.init.uniform_(m.weight, -bound, bound)
    if m.bias is not None:
      bound = 1 / math.sqrt(fan)
      nn.init.uniform_(m.bias, -bound, bound)


class SparseWeightsBase(nn.Module):
    """
    Base class for the all Sparse Weights modules
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, module, weightSparsity):
        """
        :param module:
          The module to sparsify the weights
        :param weightSparsity:
          Pct of weights that are allowed to be non-zero in the layer.
        """

        super(SparseWeightsBase, self).__init__()
        assert 0 < weightSparsity < 1

        self.module = module
        self.weightSparsity = weightSparsity
        self.register_buffer("zeroWts", self.computeIndices())
        self.rezeroWeights()

    def forward(self, x):
        if self.training:
          self.rezeroWeights()
        return self.module.forward(x)

    @abc.abstractmethod
    def computeIndices(self):
        """
        For each unit, decide which weights are going to be zero
        :return: tensor indices for all non-zero weights. See :meth:`rezeroWeights`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rezeroWeights(self):
        """
        Set the previously selected weights to zero. See :meth:`computeIndices`
        """
        raise NotImplementedError


class SparseWeights(SparseWeightsBase):
    def __init__(self, module, weightSparsity):
        """
        Enforce weight sparsity on linear module during training.
        Sample usage:
          model = nn.Linear(784, 10)
          model = SparseWeights(model, 0.4)
        :param module:
          The module to sparsify the weights
        :param weightSparsity:
          Pct of weights that are allowed to be non-zero in the layer.
        """
        super(SparseWeights, self).__init__(module, weightSparsity)

    def computeIndices(self):
        # For each unit, decide which weights are going to be zero
        outputSize, inputSize = self.module.weight.shape
        numZeros = int(round((1.0 - self.weightSparsity) * inputSize))

        outputIndices = np.arange(outputSize)
        inputIndices = np.array([np.random.permutation(inputSize)[:numZeros]
                                 for _ in outputIndices], dtype=np.long)

        # Create tensor indices for all non-zero weights
        zeroIndices = np.empty((outputSize, numZeros, 2), dtype=np.long)
        zeroIndices[:, :, 0] = outputIndices[:, None]
        zeroIndices[:, :, 1] = inputIndices
        zeroIndices = zeroIndices.reshape(-1, 2)
        return torch.from_numpy(zeroIndices.transpose())

    def rezeroWeights(self):
        zeroIdx = (self.zeroWts[0].to(torch.long), self.zeroWts[1].to(torch.long))
        self.module.weight.data[zeroIdx] = 0.0
