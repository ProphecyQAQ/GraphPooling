import argparse

def paramterParser():
    '''
    Parser paramters
    '''

    parser = argparse.ArgumentParser(description="Graph Pooling")

    parser.add_argument(
        "--dataset",
        nargs   = "?",
        default = "PROTEINS",
        help    = "Dataset Name. Default is PROTEINS",
    )
    # ENZYMES

    parser.add_argument(
        "--hidden-dim",
        type    = int,
        default = 64,
        help    = "Dimension of GNN layer1 output. Default is 32."
    )

    parser.add_argument(
        "--epochs",
        type    = int,
        default = 150,
        help    = "Number of training epochs,. Default is 100.",
    )

    parser.add_argument(
        "--batch-size",
        type    = int,
        default = 128,
        help    = "Number of graph pairs per batch. Default is 128.",
    )

    parser.add_argument(
        "--lr",
        type    = float,
        default = 0.001,
        help    = "Learing rate. Default is 0.001.",
    )

    parser.add_argument(
        "--weight-decay",
        type    = float,
        default = 5*10**-3,
        help    = "Adam weight decay. Default is 5*10^-4."
    )

    parser.add_argument(
        "--batch-norm",
        type    = bool,
        default = True,
        help    = "Use batch norm.",
    )

    parser.add_argument(
        "--num-pooling",
        type    = int,
        default = 2,
        help    = "Number of pooling operator."
    )

    parser.add_argument(
        "--assign-ratio",
        type    = float,
        default = 0.1,
        help    = "Assign ratio in pooling layer."
    )
    
    parser.add_argument(
       "--max-num-node",
       type    = int,
       default = 800,
       help    = "Max node num."
    )

    parser.add_argument(
        "--gpu-id",
        type    = str,
        default = "cuda:0",
        help    = "Set gpu id, default is 0."
    )

    return parser.parse_args()