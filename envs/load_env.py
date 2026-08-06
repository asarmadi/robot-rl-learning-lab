def load_environment(environment, method):
    if environment == 'CartPole':
        from envs.cart_pole import CartPole
        env   = CartPole(method=method)
        output_dim  = 1
        max_action  = 5
    elif environment == 'DifferentialDrive':
        from envs.differential_drive import DifferentialDrive
        env   = DifferentialDrive(method=method)
        output_dim = 2
        max_action  = 10
    return env, output_dim, max_action