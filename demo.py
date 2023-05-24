import os
import torch
import logging
from metrics.eval_function import EvalPMD
from data_load.common import Mesh
model_save_dir = 'saved_models/'
logging.basicConfig(format='%(asctime)s - [%(name)s] - %(levelname)s: %(message)s',
                    level=logging.INFO)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def demo(path_id,
         path_po,
         model:torch.nn.Module,
         model_name=None,
         path_gt=None):
    logger = logging.getLogger("demo")
    if model_name is not None and isinstance(model_name, str):
        model_path = os.path.join(model_save_dir, model_name)
        try:
            model.load_state_dict(torch.load(model_path))
            logger.info('Saved model file found and successfully read')
        except:
            logger.warning(
                'No saved model found, new model file will be created:' + str(model_path))
    model.to(device)
    model.eval()
    id = Mesh(path_id)
    po = Mesh(path_po)
    # po.add_noise(noise=0.05)          # Add noise
    v_id = id.vertices.t().unsqueeze(0).to(device)
    v_po = po.vertices.t().unsqueeze(0).to(device)
    v_pred = model(v_id, v_po)
    if path_gt is not None:
        gt = Mesh(path_gt)
        v_gt = gt.vertices.t().unsqueeze(0).to(device)
        Eval1 = EvalPMD()
        Eval1.to(device)
        eval = Eval1(v_pred, v_gt)
        logger.info('eval: %.6f' % float(eval))
        gt.save_obj('./result/gt.obj')
    v_out = v_pred[0].t().cpu().detach()
    mesh_pred = Mesh(v_out.numpy(), id.face_index)
    mesh_pred.save_obj('./result/out.obj')
    id.save_obj('./result/id.obj')
    po.save_obj('./result/po.obj')
    logger.info('finished')
    return v_out, mesh_pred

if __name__ == '__main__':
    from model import FF_PTNet
    # You can customize three mesh paths, path_gt is optional
    path_po = './datasets/smpl/id24_572.obj'
    path_id = './datasets/smpl/id26_788.obj'
    path_gt = './datasets/smpl/id26_572.obj'
    v_out, mesh_pred = demo(path_id,
                            path_po,
                            model=FF_PTNet(),
                            model_name='model_smpl.pth',
                            path_gt=path_gt)
