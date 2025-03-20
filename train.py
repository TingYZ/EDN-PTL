import torch
from networks.model_mine import highwayNet as Net
from utils import ngsimDataset, maskedNLL, maskedMSE
from torch.utils.data import DataLoader
from args import args
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random

t1 = time.time()
# 设置随机种子
seed = 2024
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True  # 启用确定性算法（禁用非确定性操作，性能略微下降）
torch.backends.cudnn.benchmark = False  # 禁用自动调优，确保每次使用相同算法


def train():

    model_type = args['model_type']
    # Initialize network
    net = Net(args)
    net = net.cuda()

    # Initialize optimizer  mse:nll = 5:3
    pretrainEpochs = 5
    trainEpochs = 5
    optimizer = torch.optim.Adam(net.parameters())
    batch_size = 128

    # Initialize data loaders
    trSet = ngsimDataset('data/TrainSet.mat', enc_size=args['dyn_size'])
    trDataloader = DataLoader(trSet, batch_size=batch_size, num_workers=8, shuffle=True, collate_fn=trSet.collate_fn)
    valSet = ngsimDataset('data/ValSet.mat', enc_size=args['dyn_size'])
    valDataloader = DataLoader(valSet, batch_size=batch_size, num_workers=8, shuffle=True, collate_fn=valSet.collate_fn)

    # 训练和测试损失，每个epoch记录一次
    train_loss = []
    val_loss = []
    writer = SummaryWriter(args['trained_model'] + f'runs/{model_type}')

    for epoch_num in range(pretrainEpochs + trainEpochs):
        if epoch_num == 0:
            print('Pre-training with MSE loss')
        elif epoch_num == pretrainEpochs:
            print('Training with NLL loss')

        """训练"""
        batch_tr_loss = []
        tr_time = 0

        for i, data in enumerate(trDataloader):
            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

            # Forward pass
            fut_pred = net(hist, nbrs, mask)
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

            # 反向传播
            optimizer.zero_grad()  # 模型参数的梯度清零
            l.backward()  # 计算梯度
            a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)  # 梯度裁剪防止梯度爆炸
            optimizer.step()  # 用优化器根据梯度更新参数

            # Track average train loss and average train time:
            batch_time = time.time()-st_time
            batch_tr_loss.append(l.item())
            tr_time += batch_time

            if i % 1000 == 999:
                print(f"step no: {i} | loss: {sum(batch_tr_loss[-1000:]) / 1000}")

        epoch_tr_loss = sum(batch_tr_loss) / len(batch_tr_loss)
        train_loss.append(epoch_tr_loss)

        """验证"""
        batch_val_loss = []
        val_time = 0

        for i, data in enumerate(valDataloader):
            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

            # Forward pass
            fut_pred = net(hist, nbrs, mask)
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

            # Track average train loss and average train time:
            batch_time = time.time()-st_time
            batch_val_loss.append(l.item())
            val_time += batch_time

        epoch_val_loss = sum(batch_val_loss) / len(batch_val_loss)
        val_loss.append(epoch_val_loss)

        # 写入训练及预测损失
        if epoch_num < pretrainEpochs:
            writer.add_scalar(tag=f'{model_type}_pre_train', scalar_value=epoch_tr_loss, global_step=epoch_num)
            writer.add_scalar(tag=f'{model_type}_pre_val', scalar_value=epoch_val_loss, global_step=epoch_num)
        else:
            writer.add_scalar(tag=f'{model_type}_train_nll', scalar_value=epoch_tr_loss, global_step=epoch_num)
            writer.add_scalar(tag=f'{model_type}_val_nll', scalar_value=epoch_val_loss, global_step=epoch_num)

        # 输出训练及预测损失
        print("Epoch no:", epoch_num,
              "| Avg train loss:", format(epoch_tr_loss, '0.4f'),
              "| Avg val loss:", format(epoch_val_loss, '0.4f'))

        # 预训练完毕，保存一次模型
        # if epoch_num == pretrainEpochs - 1:
        #     torch.save(net.state_dict(),
        #                args['trained_model'] + '{}/{}_epoch{}.tar'.format(test_name, model_type, pretrainEpochs))

    writer.close()
    torch.save(net.state_dict(),
               args['trained_model'] + '{}_epoch{}.tar'.format(model_type, pretrainEpochs + trainEpochs))
    t2 = time.time()
    print("predict consume {}".format(t2-t1))  # 100min


if __name__ == "__main__":
    # 预训练
    print("training...")
    train()


