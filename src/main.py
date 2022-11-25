from paramParse import paramterParser
from trainer import Trainer


if __name__ == '__main__':
    args = paramterParser()
    trainer = Trainer(args)
    trainer.fit()
    trainer.eval()    