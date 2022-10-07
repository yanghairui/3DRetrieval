# import torch
# import torchvision
# import torch.utils.data as Data

# class alignCollate(object):
#     def __init__(self):
#         super(alignCollate, self).__init__()
#
#     def __call__(self, batch):
#         img, _, file = zip(*batch)
#
#         return img, file
#
# def mean_std(dataSet, invalid_files):
#     mean = torch.zeros(3)
#     std  = torch.zeros(3)
#
#     toTensor = torchvision.transforms.ToTensor()
#     for (im, file) in Data.DataLoader(dataSet, collate_fn=alignCollate()):
#         im = toTensor(im[0])
#         file = file[0]
#         if file in invalid_files:
#             continue
#         for dim in range(3):
#             mean[dim] += im[dim, :, :].mean()
#             std[dim]  += im[dim, :, :].std()
#
#     return mean / len(dataSet), std / len(dataSet)
