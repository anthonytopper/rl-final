def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--history_size', type=int, default=8000, help='Replay memory size')
    parser.add_argument('--name', type=str, default='temp', help='weight file name')
    parser.add_argument('--continue_epoch', type=int, default=0, help='Continue epoch for resume the training')
    parser.add_argument('--mode', type=str, default='', help='training mode, i.e., "prioritized"')
    parser.add_argument('--delay', type=int, default=10000, help='Epoch to start apply the penalty')
    return parser
