
from dldemos.ddpm.dataset import get_dataloader

dataloader = get_dataloader(64)

'''
for batch in dataloader:
    # data, labels = batch
    # print(data.shape, labels.shape)
    # print(data)
    # print(labels)
    print(len(batch))
    print(type(batch))
    break
'''

for data, labels in dataloader:
    print(type(data))
    print(data.shape)
    print(data.shape[0])
    # print(data[0].shape)
    break