def gradient_descent_quadratic(a, b, c, x0, lr, steps):
   # initialize x as float
    x = float(x0)
    
    # perform gradient descent updates
    for _ in range(steps):
        gradient = 2 * a * x + b   # derivative f'(x)
        x = x - lr * gradient      # update rule
    
    return float(x)