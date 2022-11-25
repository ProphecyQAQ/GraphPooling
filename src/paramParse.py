import argparse

def paramterParser():
    '''
    Parser paramters
    '''

    parser = argparse.ArgumentParser(description="Graph Pooling")

    parser.add_argument(
        "--dataset",
        nargs   = "?",
        default = "ENZYMES",
        help    = "Dataset Name. Default is PROTEINS",
    )
    # ENZYMES


    parser.add_argument(
        "--filters-1",
        type    = int,
        default = 128,
        help    = "Dimension of GNN layer1 output. Default is 64."
    )


    parser.add_argument(
        "--filters-2",
        type    = int,
        default = 64,
        help    = "Dimension of GNN layer1 output. Default is 32."
    )

    parser.add_argument(
        "--filters-3",
        type    = int,
        default = 32,
        help    = "Dimension of GNN layer1 output, default is 16. Default is 16."
    )

    parser.add_argument(
        "--epochs",
        type    = int,
        default = 100,
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
        default = 5*10**-4,
        help    = "Adam weight decay. Default is 5*10^-4."
    )

    parser.add_argument(
        "--batch-norm",
        type    = bool,
        default = True,
        help    = "Use batch norm.",
    )

    parser.add_argument(
        "--gpu-id",
        type    = str,
        default = "cuda:0",
        help    = "Set gpu id, default is 0."
    )

    return parser.parse_args()