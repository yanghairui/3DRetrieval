import numpy as np

class PerClassAccuracy(object):
    def __init__(self, num_classes):
        super(PerClassAccuracy, self).__init__()
        self.num_classes = num_classes

        self.classes = {}
        for class_idx in range(num_classes):
            self.classes[class_idx] = {
                'correct': 0,
                'total': 0
            }

    def reset(self):
        for class_idx in range(self.num_classes):
            self.classes[class_idx]['correct'] = 0
            self.classes[class_idx]['total'] = 0

    def update(self, target, predict):
        for idx in range(target.shape[0]):
            self.classes[int(target[idx])]['total'] += 1

        eq_idx = np.where(target == predict)[0]
        for idx in range(eq_idx.shape[0]):
            self.classes[int(target[eq_idx[idx]])]['correct'] += 1

    def calc(self):
        accu_perclass = []
        correct, total = 0, 0

        mAP = 0.

        true_count = 0
        for key in self.classes.keys():
            correct += self.classes[key]['correct']
            total += self.classes[key]['total']

            if self.classes[key]['total'] == 0:
                accu_perclass.append(0.)
                continue
            true_count += 1
            mAP += float(self.classes[key]['correct']) / self.classes[key]['total']

            accu_perclass.append(
                float(self.classes[key]['correct']) / self.classes[key]['total'])

        return accu_perclass, float(correct) / total, float(mAP) / true_count
