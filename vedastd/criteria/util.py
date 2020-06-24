import torch
import numpy as np


def ohem_single(pred_text, gt_text, mask):
    pos_num = torch.sum(gt_text > 0.5) - torch.sum((gt_text > 0.5) & (mask <= 0.5))

    if pos_num == 0:
        selected_mask = mask
        return torch.unsqueeze(selected_mask, dim=0).float()

    neg_num = torch.sum(gt_text <= 0.5)
    neg_num = min(pos_num * 3, neg_num)

    if neg_num == 0:
        selected_mask = mask
        return torch.unsqueeze(selected_mask, dim=0).float()

    neg_score = pred_text[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((pred_text >= threshold) | (gt_text > 0.5)) & (mask > 0.5)
    return torch.unsqueeze(selected_mask, dim=0).float()


def ohem_single_bk(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()

    return selected_masks


if __name__ == '__main__':
    torch.manual_seed(1)
    score = torch.rand(224, 224)
    gt_text = torch.rand(224, 224)
    training_mask = torch.rand(224, 224)
    print(torch.sum(ohem_single(score, gt_text, training_mask)))

    score_np = score.numpy()
    gt_text_np = gt_text.numpy()
    training_mask_np = training_mask.numpy()
    print(np.sum(ohem_single_bk(score_np,gt_text_np,training_mask_np)))
