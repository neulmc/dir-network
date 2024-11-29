import numpy as np
import cv2
import os
import sklearn.metrics as sk
import matplotlib.pyplot as plt

# return I(x,y) = H(x) - H(x|y), higher MI means higher correlation;
# uncertainty could be any set
# the bigger, the better
def compute_mi(pred, gt, uncertainty, threshold = 0.5):
    """Computes mutual information between error and uncertainty.

    Args:
        pred: numpy array [0-1].
        gt: numpy binary array {0,1}.
        threshold: convert pred to binary
        uncertainty: numpy float array indicating uncertainty.

    Returns:
        mutual_information
    """
    #pred[pred > threshold] = 1
    #pred[pred <= threshold] = 0
    error = 1 * ~np.equal(pred, gt)

    hist_2d, x_edges, y_edges = np.histogram2d(error.ravel(), uncertainty.ravel())

    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

# return dice score based on error and uncertainty, needs a threshold
# uncertainty might be [0-1], or select a proper threshold
# the bigger, the better
def compute_overlap(pred, gt, uncertainty, threshold = 0.5, bin = 20):
    """Computes overlap between error and uncertainty.

    Args:
        pred: numpy array [0-1].
        gt: numpy binary array {0,1}.
        threshold: convert pred to binary
        uncertainty: numpy float array indicating uncertainty.
        bin: IoU bins to find larger one

    Returns:
        dice overlap.
    """
    #pred[pred > threshold] = 1
    #pred[pred <= threshold] = 0
    error = 1 * ~np.equal(pred, gt)
    r = 1 * np.equal(pred, gt)
    e_dice = 0
    r_dice = 0

    for i in range(bin):
        uncertainty_threshold = i / bin
        uncertainty_ = (uncertainty > uncertainty_threshold).astype(int)
        intersection = np.logical_and(error, uncertainty_)
        e_dice_ = 2.0 * intersection.sum() / (error.sum() + uncertainty_.sum())
        e_dice = max(e_dice_, e_dice)

        intersection = np.logical_and(r, 1 - uncertainty_)
        r_dice_ = 2.0 * intersection.sum() / (r.sum() + (1 - uncertainty_).sum())
        r_dice = max(r_dice_, r_dice)

    return e_dice, r_dice

# return calibration score based on error and uncertainty,
# uncertainty must be [0-1]
# pred is the probability from network, threshold is related to pred
# the smaller, the better
def compute_calibration(pred, gt, uncertainty, prob, threshold = 0.5, nb_bins = 20):
    confidences = prob.max(axis=2)
    #confidences = 1 - uncertainty

    nb_bins = nb_bins
    bin_boundaries = np.linspace(0, 1, nb_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    #pred[pred > threshold] = 1
    #pred[pred <= threshold] = 0
    accuracies = np.equal(pred, gt).astype(int)
    not_bg = pred != 0

    confidences = confidences[not_bg]
    accuracies = accuracies[not_bg]

    ece = 0
    bins_avg_conf = []
    bins_avg_acc = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.greater(confidences, bin_lower) * np.less(confidences, bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bins_avg_conf.append(avg_confidence_in_bin)
            bins_avg_acc.append(accuracy_in_bin)
    return ece

# return auroc
# confidence = 1 - uncertainty
def compute_auroc(pred, gt, uncertainty, threshold = 0.5):
    #pred[pred > threshold] = 1
    #pred[pred <= threshold] = 0
    labels = 1 * np.equal(pred, gt)
    mPu = np.mean(uncertainty[labels])
    mNu = np.mean(uncertainty[1 - labels])
    labels = labels.reshape([labels.shape[0] * labels.shape[1]])
    uncertainty = uncertainty.reshape([uncertainty.shape[0] * uncertainty.shape[1]])
    AUROC = sk.roc_auc_score(labels, 1 - uncertainty)
    AUPR = sk.average_precision_score(labels, 1 - uncertainty)
    return AUROC, AUPR, mPu, mNu


# output accurate; dice score
def compute_accurate(pred, gt, threshold = 0.5):
    acc_ = 1 * np.equal(pred, gt)
    acc = np.mean(acc_)
    dices_fenzi = []
    dices_fenmu = []
    for class_i in range(3):
        pred_tmp = np.zeros(pred.shape)
        gt_tmp = np.zeros(gt.shape)
        pred_tmp[pred == (class_i + 1)] = 1
        gt_tmp[gt == (class_i + 1)] = 1
        intersection = np.logical_and(pred_tmp, gt_tmp)
        dices_fenzi.append(2.0 * intersection.sum())
        dices_fenmu.append(pred_tmp.sum() + gt_tmp.sum())
    return acc, dices_fenzi, dices_fenmu

# post procession
def post(output, uncertainty_image, pad_area=20, uncer_t=0.8):
    # output: [0-1]
    # uncertainty_image [0-1]

    class Point(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def getX(self):
            return self.x

        def getY(self):
            return self.y

    def convertbitmap(output):
        output[output > 0.5] = 255
        output[output <= 0.5] = 0
        return output

    def getGrayDiff(img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))

    def selectConnects():
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return connects

    def regionGrow(img, seeds, label):
        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        connects = selectConnects()
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)

            seedMark[currentPoint.x, currentPoint.y] = label
            for i in range(4):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                if grayDiff < 1 and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = label
                    seedList.append(Point(tmpX, tmpY))
        return seedMark

    def pad_small_hole(output, uncertainty_image, pad_area=20, uncer_t=0.8):
        # pad_area: max pad area
        conf_image = 1 - uncertainty_image
        seedmarks = np.zeros(output.shape)
        idx_x, idx_y = np.where(output == 0)
        label = 1
        for i in range(len(idx_x)):
            tmp_x = idx_x[i]
            tmp_y = idx_y[i]
            if seedmarks[tmp_x, tmp_y] != 0:
                continue
            else:
                seedmarks_tmp = regionGrow(output, seeds=[Point(tmp_x, tmp_y)], label=label)
                seedmarks[seedmarks_tmp == label] = label
                label += 1
        max_size = 0
        max_size_label = 0
        for i in range(label):
            if i == 0:
                continue
            area_tmp = len(np.where(seedmarks == i)[0])
            if area_tmp > max_size:
                max_size = area_tmp
                max_size_label = i
                # confidence
            if area_tmp < pad_area:
                if np.mean(conf_image[np.where(seedmarks == i)]) > uncer_t:
                    continue
                output[np.where(seedmarks == i)] = 255
                seedmarks[np.where(seedmarks == i)] = 0
        return output, seedmarks, max_size_label

    # convert bimap
    output = convertbitmap(output)
    # pad small inner hole
    output, back_seedmarks, max_size_label = pad_small_hole(output, uncertainty_image, pad_area=pad_area,
                                                            uncer_t=uncer_t)

    return output

# measurment
def measurement(output, gt, uncertainty, filt_measure = 20, threshold = 0.5, uncertain_threshold = 0.8, uncertain_fixed = False):
    # uncertain_threshold: a threshold used for filt out area
    # bigger means loose, smaller means tighter

    def hist(area_list):
        bin_boundaries = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        prob_in_bins = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(area_list, bin_lower) * np.less(area_list, bin_upper)
            prob_in_bins.append(in_bin.mean())
        prob_in_bins.append(np.greater(area_list, bin_boundaries[-1]).mean())
        prob_in_bins = np.array(prob_in_bins)
        return prob_in_bins, bin_boundaries

    def kl_diverge(prob_p, prob_q):
        eps = 1e-8
        kl = - prob_p * np.log((prob_q + eps) / (prob_p + eps))
        return kl.sum()

    class Point(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def getX(self):
            return self.x

        def getY(self):
            return self.y

    def getGrayDiff(img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))

    def selectConnects():
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return connects

    def regionGrow(img, seeds, label):
        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        connects = selectConnects()
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)

            seedMark[currentPoint.x, currentPoint.y] = label
            for i in range(4):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                if grayDiff < 1 and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = label
                    seedList.append(Point(tmpX, tmpY))
        return seedMark

    def area_get(output, uncertainty):
        area = []
        uncer = []
        seedmarks = np.zeros(output.shape)
        idx_x, idx_y = np.where(output == 255)
        label = 1
        for i in range(len(idx_x)):
            tmp_x = idx_x[i]
            tmp_y = idx_y[i]
            if seedmarks[tmp_x, tmp_y] != 0:
                continue
            else:
                seedmarks_tmp = regionGrow(output, seeds=[Point(tmp_x, tmp_y)], label=label)
                seedmarks[seedmarks_tmp == label] = label
                label += 1
        for i in range(label):
            if i == 0:
                continue
            area_tmp = len(np.where(seedmarks == i)[0])
            area.append(area_tmp)
            uncer.append(np.mean(uncertainty[seedmarks == i]))
        area = np.array(area)
        uncer = np.array(uncer)
        return area, uncer

    output[output > threshold] = 1
    output[output <= threshold] = 0
    output_area, output_uncer = area_get(output * 255, uncertainty)
    gt_area, gt_uncer = area_get(gt * 255, uncertainty)

    output_area_filt = output_area[output_area > filt_measure]
    output_uncer_filt = output_uncer[output_area > filt_measure]
    # uncertain fixed
    if uncertain_fixed:
        output_area_filt = output_area_filt[output_uncer_filt < uncertain_threshold]

    gt_area_filt = gt_area[gt_area > filt_measure]
    mean_dis = np.abs(np.mean(output_area_filt) - np.mean(gt_area_filt))
    std_dis = np.abs(np.std(output_area_filt) - np.std(gt_area_filt))

    output_list_prob, tick_label = hist(output_area_filt)
    gt_list_prob, _ = hist(gt_area_filt)
    kl_dis = kl_diverge(gt_list_prob, output_list_prob)

    return mean_dis, std_dis, kl_dis

if __name__ == '__main__':
    gt = np.random.random([100,1])
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0
    pred = np.random.random([100,1])
    uncertainty = np.random.random([100,1])
    mi = compute_mi(pred, gt, uncertainty)
    undice = compute_overlap(pred, gt, uncertainty)
    ece = compute_calibration(pred, gt, uncertainty)
    acc, dice = compute_accurate(pred, gt)
    print(mi)
    print(undice)
    print(ece)
    print(acc)
    print(dice)
    print('over')