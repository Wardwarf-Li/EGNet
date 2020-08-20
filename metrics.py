import numpy as np
import torch
import torchvision.transforms as transforms


# self-adaptive threshold : T = mean + std
def fbeta_score(preds, gts, beta2):
    thresholds = []
    Fmeasures = []

    for i, smap in enumerate(preds):
        T = smap.mean() + smap.std()
        thresholds.append(T.item())

    for i in range(len(thresholds)):
        temp = (preds[i] >= thresholds[i]).float()
        tp = (temp * gts[i]).sum()
        precision, recall = tp / temp.sum(), tp / gts[i].sum()
        Fm = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        Fm[Fm != Fm] = 0 # for Nan
        Fmeasures.append(Fm.item())

    # print(Fmeasures)
    return np.mean(Fmeasures)


# all thresholds
def Fmeasure(preds, gts, beta2=0.3):
    avg_p, avg_r, img_num = 0.0, 0.0, 0.0

    for i in range(preds.size()[0]):
        precision, recall = eval_pr(preds[i], gts[i], 255)
        avg_p += precision
        avg_r += recall
        img_num += 1.0
    avg_p /= img_num
    avg_r /= img_num
    Fm = (1 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)
    Fm[Fm != Fm] = 0  # for Nan

    return Fm.max().item()


def eval_pr(pred, gt, num):
    prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
    thlist = torch.linspace(0, 1 - 1e-10, num).cuda()

    for i in range(num):
        temp = (pred >= thlist[i]).float()
        tp = (temp * gt).sum()
        prec[i], recall[i] = tp / (temp.sum() + 1e-20), tp / (gt.sum() + 1e-20)
    return prec, recall


def mae(preds, gts):
    return torch.abs(preds - gts).mean().item()


def Smeasure(preds, gts):
    alpha, avg_q, img_num = 0.5, 0.0, 0.0
    for i in range(preds.size()[0]):
        y = gts[i].mean()
        if y == 0:
            x = preds[i].mean()
            Q = 1.0 - x
        elif y == 1:
            x = preds[i].mean()
            Q = x
        else:
            Q = alpha * S_object(preds[i], gts[i]) + (1 - alpha) * S_region(preds[i], gts[i])
            if Q.item() < 0:
                Q = torch.tensor([0.0])

        img_num += 1.0
        avg_q += Q.item()

    avg_q /= img_num
    return avg_q


def S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q


def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score


def S_region(pred, gt):
    X, Y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divideGT(gt, X, Y)
    p1, p2, p3, p4 = dividePrediction(pred, X, Y)
    Q1 = ssim(p1, gt1)
    Q2 = ssim(p2, gt2)
    Q3 = ssim(p3, gt3)
    Q4 = ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    return Q


def centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = torch.eye(1).cuda() * round(cols / 2)
        Y = torch.eye(1).cuda() * round(rows / 2)
    else:
        total = gt.sum()
        i = torch.from_numpy(np.arange(0, cols)).cuda().float()
        j = torch.from_numpy(np.arange(0, rows)).cuda().float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)
    return X.long(), Y.long()


def divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


def ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q





