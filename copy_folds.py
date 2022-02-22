import os
import shutil

if __name__ == "__main__":

    for data_name in ['MO', 'TNBC']:
        for i in range(10):
            src = f"/Users/admin/AI_group/heqian/biomedical/data/{data_name}10folds/{data_name}{i}/train_val_test.json"
            dst = f"/Users/admin/PycharmProjects/BioMedSeg/ISBI/data/{data_name}/fold{i}.json"
            shutil.copyfile(src, dst)