from tqdm import tqdm
import torch
import config as cfg
from dataset import get_loader
from model import get_model_loss_optimizer
from torch.utils.tensorboard import SummaryWriter


def train(model, loader, optimizer, criterion, device):

    total_loss = 0

    model.train()
    for i, data in tqdm(enumerate(loader)):

        x1 = data[0].to(device)
        x2 = data[1].to(device)
        x3 = data[2].to(device)

        y1 = data[3].to(device)
        y2 = data[4].to(device)
        y3 = data[5].to(device)

        model.zero_grad()
        pred1, pred2, pred3 = model(x2, x3, x1)

        loss = criterion(pred1, pred2, pred3, y1, y2, y3)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('Total Loss:', total_loss)

    return total_loss


def validate(model, loader, criterion, device):

    total_loss = 0

    model.eval()

    with torch.no_grad():

        for i, data in tqdm(enumerate(loader)):

            x1 = data[0].to(device)
            x2 = data[1].to(device)
            x3 = data[2].to(device)

            y1 = data[3].to(device)
            y2 = data[4].to(device)
            y3 = data[5].to(device)

            pred1, pred2, pred3 = model(x2, x3, x1)

            loss = criterion(pred1, pred2, pred3, y1, y2, y3)
            total_loss += loss.item()

    print('Valid loss: ', total_loss)

    return total_loss


def main(cfg):

    train_loader = get_loader(cfg.train_path, cfg.max_len, cfg.batch_size, cfg.num_worker, cfg.shuffle)
    val_loader = get_loader(cfg.val_path, cfg.max_len, cfg.batch_size, cfg.num_worker, cfg.shuffle)

    model, criterion, optimizer = get_model_loss_optimizer(lr=cfg.lr, device=cfg.device)
    writer = SummaryWriter()

    prev_loss = 1e7
    for i in range(cfg.epochs):

        print("Epoch: {}".format(i+1))

        total_train_loss = train(model, train_loader, optimizer, criterion, cfg.device)
        writer.add_scalar("Loss/train", total_train_loss, i)

        total_val_loss = validate(model, val_loader, criterion, cfg.device)
        writer.add_scalar("Loss/val", total_val_loss, i)


        if total_val_loss < prev_loss:

            checkpoint = {
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()
            }

            torch.save(checkpoint, cfg.model_path + 'model.pth')
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main(cfg)
