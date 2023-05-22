import re

if __name__ == '__main__':
    import torch
    from torchmetrics import CharErrorRate

    # 准备模型的输出和标签数据
    outputs = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.8, 0.1, 0.0, 0.1], [0.4, 0.3, 0.2, 0.1]])
    targets = ['cat', 'dog', 'bird']

    # 初始化 CharErrorRate 对象
    cer = CharErrorRate(ignore_case=True)

    # 计算 CER
    cer.update(outputs, targets)
    cer = cer.compute()

    print(f'Character Error Rate: {cer:.4f}')
