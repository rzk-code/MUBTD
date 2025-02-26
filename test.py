# external imports
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as Data
# internal imports
from Model.config import args
from FusionNet.data_generator2 import Dataset
from FusionNet.FusionNet import FusionNet
from Utils.losses import CrossEntropyLoss, DisentanglementLoss




def count_parameters(model):  # 统计模型包含的参数个数
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())  # 筛选出需要求导的参数
    params = sum([np.prod(p.size()) for p in model_parameters])  # 计算每层的参数总数，并求和得到整个模型的参数
    return params


def test(paramPath, alpha, tau, seed, datasetType):
    # param为str，保存的参数文件名
    # 创建需要的文件夹并指定gpu
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 导入模型参数
    model = FusionNet(3).to(device)

    ckp = torch.load(paramPath)
    model.load_state_dict(ckp['model'])

    model.eval()

    # 测试循环
    # 导入数据集
    benign_npz  =  np.load('')
    malignant_npz  =  np.load('')

    DS = Dataset(benign_npz, malignant_npz, datasetType, 0, seed=seed)
    print(datasetType + ':', len(DS))

    DL = Data.DataLoader(DS, batch_size=32, num_workers=0, shuffle=False, drop_last=False)
    right = 0
    test_loss = 0

    pred = []
    true = []

    for image, feature_T, target in DL:
        # 数据格式转换
        image = image.to(device).float()
        feature_T = feature_T.to(device).float()
        target = target.to(device).float()

        # 前向传播
        output, feature_B, feature_D = model(image, feature_T)

        pred.append(output)
        true.append(target)

        # 计算损失函数
        loss_CE = CrossEntropyLoss(output, target)
        loss_CL = DisentanglementLoss(feature_B, feature_D, tau)

        loss = loss_CE + alpha * loss_CL
        test_loss += loss.item()

        # 计算准确率
        output = F.softmax(output, dim=-1)
        top_pred = output.argmax(1)
        target = target.argmax(1)
        right += torch.sum(top_pred == target)

        # tensor的item方法可以将单元素tensor对象转换为python标量
        print("loss: %f" % (loss.item()), flush=True)



