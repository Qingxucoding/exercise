def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """

    parser.add_argument('--env_name', default="CartPole-v0", help='environment name') # 环境名称
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--buffer_size", default=1e4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.0001, type=float) # 学习率
    parser.add_argument("--gamma", default=0.99, type=float) # 折扣因子
    parser.add_argument("--grad_norm_clip", default=10, type=float) # 进行梯度裁剪，防止梯度爆炸
    parser.add_argument("--test", default=False, type=bool) # 是否用测试模式
    parser.add_argument("--use_cuda", default=False, type=bool) # 是否用GPU训练
    parser.add_argument("--n_episodes", default=2000, type=int)  # 迭代次数
    parser.add_argument("--learning_freq", default=1, type=int) # 学习频率
    parser.add_argument("--target_update_freq", default=1000, type=int) # 更新目标网络的频率

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_episodes", default=int(1000), type=int)

    return parser
