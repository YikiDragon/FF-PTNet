import logging
import os

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_load.smpl import SMPLDataset
from metrics.eval_function import EvalPMD

model_save_dir = 'saved_models/'
logging.basicConfig(format='%(asctime)s - [%(name)s] - %(levelname)s: %(message)s',
                    level=logging.INFO)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def eval(eval_dataset,
         model: torch.nn.Module,
         model_name=None,
         batch_size=8):
    logger = logging.getLogger("EvalMode")
    if model_name is not None and isinstance(model_name, str):
        save_path = os.path.join(model_save_dir, model_name)
        try:
            model.load_state_dict(torch.load(save_path))
            logger.info('Saved model file found and successfully read')
        except:
            logger.warning(
                'No saved model found, new model file will be created:' + str(save_path))
    model.to(device)
    model.eval()
    dataloader = DataLoader(eval_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=10,
                            pin_memory=True,
                            drop_last=False)
    Eval1 = EvalPMD()
    Eval1.to(device)
    logger.info('Start of evaluating......')
    eval_mean = 0
    with tqdm(total=len(dataloader.dataset)) as eval_bar:
        eval_bar.set_description_str('Evaluating')
        for id, po, gt in dataloader:
            v_id = id.v.to(device)
            # e_id = id.edge_index.to(device)
            b_id = id.batch.to(device)
            v_po = po.v.to(device)
            # e_po = po.edge_index.to(device)
            b_po = po.batch.to(device)
            v_gt = gt.v.to(device)
            with torch.no_grad():
                v_id = v_id.view(b_id.max()+1, -1, 3).permute(0, 2, 1)
                v_po = v_po.view(b_po.max()+1, -1, 3).permute(0, 2, 1)
                v_gt = v_gt.view(b_po.max()+1, -1, 3).permute(0, 2, 1)
                v_pred = model(v_id, v_po)
                eval = Eval1(v_pred, v_gt)
                eval_mean += eval
            eval_bar.write(f"EvalData: {eval_bar.n // batch_size} Eval: {eval:>7f}")
            eval_bar.set_postfix(eval=float(eval))
            eval_bar.update(dataloader.batch_size)
        eval_bar.reset()
        logger.info('evaluation completed')
        eval_mean = eval_mean / len(dataloader)
    model.train()
    return eval_mean


if __name__ == '__main__':
    from model import FF_PTNet
    NPTDS_eval = SMPLDataset(path='./datasets/smpl/',
                             identity_num=30,
                             pose_num=800,
                             identity_range=(16, 29),
                             pose_range=(400, 799),
                             shuffle_points=True,
                             type='obj')
    eval_avg = eval(eval_dataset=NPTDS_eval,
                    model=FF_PTNet(),
                    model_name='model_smpl.pth')
    print("eval_avg: %f" % eval_avg)

