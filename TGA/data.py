#from load_duf import DataloadFromFolder
from load_train import DataloadFromFolder
from load_test import DataloadFromFolderTest
from torchvision.transforms import Compose, ToTensor

def transform():
    return Compose([
             ToTensor(),
            ])

def get_training_set(data_dir, upscale_factor, data_augmentation, file_list):
    return DataloadFromFolder(data_dir, upscale_factor, data_augmentation, file_list, transform=transform())

#def get_eval_set(data_dir, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, transform=transform())
#    return DatasetFromValFolder(data_dir, upscale_factor, data_augmentation, file_list, patch_size, transform=transform())

def get_test_set(data_dir, file_list, upscale_factor, nFrames):
    return DataloadFromFolderTest(data_dir, file_list, upscale_factor, nFrames, transform=transform())

