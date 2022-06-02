import torch


def stage2_pred(device, stage2, pred_angle, pred_angle_attack, sample, y_true, y_pred, y_pred_attack):
    # pass results to stage 2
    batch_x, angle, anomaly = sample
    batch_x = batch_x.type(torch.FloatTensor)
    angle = angle.type(torch.FloatTensor)
    anomaly = anomaly.type(torch.int16)
    batch_x = batch_x.to(device)
    angle = angle.to(device)
    anomaly = anomaly.to(device)

    y_true.extend(anomaly.data)

    output = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_angle)))
    values, indices = torch.max(output, dim=1)
    y_pred.extend(indices.data)

    output_advGAN = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_angle_attack)))
    values, indices = torch.max(output_advGAN, dim=1)
    y_pred_attack.extend(indices.data)
