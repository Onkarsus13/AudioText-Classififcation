import torchmetrics
import torch
from model import get_model_loss_optimizer
from dataset import get_loader
import config as cfg
from tqdm import tqdm


def f1(cfg):

    device = cfg.device

    test_loader = get_loader(cfg.test_path, cfg.max_len, cfg.batch_size, cfg.num_worker, cfg.shuffle)

    model, _, _ = get_model_loss_optimizer(cfg.lr, cfg.device)
    model.load_state_dict(torch.load(cfg.model_path + 'model.pth')['state_dict'])
    print('model is loaded ...')

    final_f1 = []

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):

            x1 = data[0].to(device)
            x2 = data[1].to(device)
            x3 = data[2].to(device)

            y1 = data[3].to(device)
            y2 = data[4].to(device)
            y3 = data[5].to(device)

            pred1, pred2, pred3 = model(x2, x3, x1)

            f1_1 = torchmetrics.F1Score(num_classes=pred1.shape[-1]).to(device)
            f1_1 = f1_1(pred1, y1)
            f1_2 = torchmetrics.F1Score(num_classes=pred2.shape[-1]).to(device)
            f1_2 = f1_2(pred2, y2)
            f1_3 = torchmetrics.F1Score(num_classes=pred3.shape[-1]).to(device)
            f1_3 = f1_3(pred3, y3)

            f1 = (f1_1 + f1_2 + f1_3)/3

            final_f1.append(f1)

    
    return (sum(final_f1) / len(final_f1))


if __name__ == "__main__":
    
    print('Test F1 ', f1(cfg).item())
