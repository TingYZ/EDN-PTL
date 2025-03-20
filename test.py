import torch
from torch.utils.data import DataLoader
from networks.model_mine import highwayNet
from utils import ngsimDataset, maskedMSETest
from args import args
from tqdm import tqdm
import scipy.io as sio

# 设置随机种子
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True  # 启用确定性算法（禁用非确定性操作，性能略微下降）
torch.backends.cudnn.benchmark = False  # 禁用自动调优，确保每次使用相同算法


def pred_compute(net, test_dataloader):
    """
    在测试集上测试当前网络效果
    """
    loss = torch.zeros(25).cuda()
    counts = torch.zeros(25).cuda()

    for i, data in tqdm(enumerate(test_dataloader)):
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
        if nbrs.shape[1] == 0:
            continue

        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()

        # Forward pass
        fut_pred = net(hist, nbrs, mask)
        l, c = maskedMSETest(fut_pred, fut, op_mask)

        # Track average train loss:
        loss += l.detach()  # tensor(25)
        counts += c.detach()  # tensor(25)

    #
    rmse = torch.pow(loss / counts, 0.5).cpu().numpy()[4::5] * 0.3048  # 求每个时刻的均方根 array(25,)
    ade = torch.mean(torch.pow(loss / counts, 0.5)).item() * 0.3048  # ADE
    fde = torch.pow(loss / counts, 0.5)[-1].item() * 0.3048 # FDE

    return [rmse, ade, fde]


def pred():
    """
    准备模型和测试集
    """
    model_type = args['model_type']
    test_set = ngsimDataset(f'data/TestSet.mat', enc_size=args['dyn_size'])
    test_dataloader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=8,
                                 collate_fn=test_set.collate_fn)

    net = highwayNet(args)
    net.load_state_dict(torch.load(args['trained_model'] + f'{model_type}_epoch5.tar'))
    net.cuda()

    test_loss = pred_compute(net, test_dataloader)

    return test_loss


if __name__ == "__main__":

    error = pred()

    print(error)