import torch


def loss_calculate(x_oh, label_oh, loss_fn):
    """
    This is the loss for one-hot vector.
    """
    loss = 0.
    for i, prediction in enumerate(x_oh, 0):
        loss += loss_fn(prediction, label_oh) * torch.log(torch.Tensor([i+1]))
    return loss

def ml_loss_single(prediction, y, h_com):
    """this is the ML(Maximum likelihood) Loss without weights:"""
    loss_fn = torch.nn.MSELoss().cuda()
    loss = loss_fn(y, torch.bmm(h_com, prediction.view(y.shape[0], -1, 1)).squeeze(-1)).cpu()
    return loss


def common_loss(prediction, labels, rate):
    """
    This is the MSE loss; The input should be only an array, not a list.
    """
    loss_fn = torch.nn.MSELoss()
    modulated_labels = (2 * labels - 2**rate + 1).to(torch.float32)
    loss = loss_fn(prediction, modulated_labels)
    return loss


def weighted_mse(predictions, labels, rate):
    """
    The weighted MSE loss; The input should be a list, each element of which contains the result for one iteration.
    """
    loss_fn = torch.nn.MSELoss()
    modulated_labels = (2 * labels - 2**rate + 1).to(torch.float32)
    loss = 0
    for i, prediction in enumerate(predictions, 0):
        loss += loss_fn(prediction, modulated_labels) * torch.log(torch.Tensor([i+1]))
    return loss


def weighted_cross_entropy(probs, labels):
    """
    The weighted Cross-entropy loss; The input should be a list, each element of which contains the result for one iteration.
    """
    loss_fn = torch.nn.NLLLoss()
    loss = 0
    for i, prob in enumerate(probs, 0):
        loss += loss_fn(torch.log(prob.permute(0, 2, 1)), labels) * torch.log(torch.Tensor([i+1]))
    return loss


def cross_entropy_distance(predictions, labels, rate):
    """
    Proposed equivalent loss; The input should be a list, each element of which contains the result for one iteration.
    """
    loss_fn = torch.nn.NLLLoss()
    softmax = torch.nn.Softmax(dim=-1)
    length = 2 ** rate
    symbols = torch.linspace(1 - length, length - 1, length)
    # modulated_labels = (2 * labels - 2**rate + 1).to(torch.float32)
    loss = 0
    # sigma = 2
    for i, prediction in enumerate(predictions, 0):
        dis = prediction.unsqueeze(-1).repeat(1, 1, length) - symbols.unsqueeze(0).unsqueeze(0).repeat(20, 32, 1)
        likelihood = torch.exp(-torch.square(dis))
        # likelihood = -torch.square(dis) / sigma

        likelihood_max, _ = torch.max(likelihood.permute(0, 2, 1), dim=1)
        # log_max, _ = torch.max(likelihood_tilde.permute(1, 2, 0), dim=1)
        likelihood_max = likelihood_max.unsqueeze(1).repeat([1, length, 1])

        loss += loss_fn(torch.log(softmax(likelihood.permute(0, 2, 1) - likelihood_max)), labels) * torch.log(torch.Tensor([i+1]))
    return loss

