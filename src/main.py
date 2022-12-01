from paramParse import paramterParser
from trainer import Trainer
from utils import setup_seed


if __name__ == '__main__':
    setup_seed(2022)
    args = paramterParser()
    trainer = Trainer(args)
    trainer.fit()
    print("Test acc is {}".format(trainer.eval(validate=False)))