import os
from munch import Munch
import continual_train

def main():
    with open('config.yaml') as f:
        opt = Munch.fromYAML(f)
    
    gpus = ','.join([str(i) for i in opt.gpu])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    continual_train.train_and_evaluate(opt)

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()