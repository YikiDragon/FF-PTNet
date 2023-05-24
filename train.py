import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.optim as optim
import torch_geometric.data
from torch_geometric.loader import DataLoader
import logging
from tqdm import tqdm
from metrics.loss_function import LossRecL2, LossEdge

model_save_dir = 'saved_models/'
logging.basicConfig(format='%(asctime)s - [%(name)s] - %(levelname)s: %(message)s',
                    level=logging.INFO)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_simple_dataset(model: torch.nn.Module,
                         train_dataset: torch_geometric.data.Dataset,
                         saved_name='model.pth',
                         lr=0.001,
                         batch_size=4,
                         epoch_num=8,
                         shuffle=False,
                         num_workers=11):
    logger = logging.getLogger("TrainMode")
    save_path = os.path.join(model_save_dir, saved_name)
    try:
        model.load_state_dict(torch.load(save_path))
        logger.info('Saved model file found and successfully read')
    except:
        logger.warning(
            'No saved model found, new model file will be created:' + str(save_path))
    model.to(device)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=False)

    loss1 = LossRecL2()
    loss2 = LossEdge()
    loss1.to(device)
    loss2.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)

    logger.info('Start of training......')
    epoch_bar = tqdm(range(epoch_num))
    epoch_bar.set_description_str('Training')
    for epoch in epoch_bar:
        loss_mean = 0
        with tqdm(total=len(train_dataloader.dataset)) as batch_bar:
            batch_bar.set_description_str('Epoch %d' % epoch)
            for id, po, gt in train_dataloader:
                v_id = id.v.to(device)
                e_id = id.edge_index.to(device)
                v_po = po.v.to(device)
                # e_po = po.edge_index.to(device)
                v_gt = gt.v.to(device)
                # e_gt = gt.edge_index.to(device)
                v_id = v_id.view(batch_size, -1, 3).permute(0, 2, 1)
                v_po = v_po.view(batch_size, -1, 3).permute(0, 2, 1)
                v_pred = model(v_id, v_po)          # B, 3, N
                v_pred = v_pred.permute(0, 2, 1).reshape(-1, 3)
                v_id = v_id.permute(0, 2, 1).reshape(-1, 3)
                loss = loss1(v_pred, v_gt) + 5e-4 * loss2(v_pred, e_id, v_id, e_id)
                loss_mean += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_bar.write(f"batch: {batch_bar.n // train_dataloader.batch_size} loss: {loss:>7f}")
                batch_bar.set_postfix(loss=float(loss), lr=float(optimizer.param_groups[0]['lr']))
                batch_bar.update(train_dataloader.batch_size)
            batch_bar.reset()
            torch.save(model.state_dict(), save_path)
            scheduler.step()
        logger.info('Model saved to:' + str(save_path))
    logger.info('Train completed!')
    return model


if __name__ == '__main__':
    from model import FF_PTNet
    from data_load.smpl import SMPLDataset
    NPTDS_train = SMPLDataset(path='./datasets/smpl/',
                              identity_num=30,
                              pose_num=800,
                              identity_range=(0, 15),
                              pose_range=(0, 399),
                              shuffle_points=True,
                              type='obj')
    train_simple_dataset(model=FF_PTNet(),
                         train_dataset=NPTDS_train,
                         saved_name='model_smpl.pth',
                         lr=0.001,
                         batch_size=8,
                         epoch_num=200)
