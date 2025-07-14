def generate_intervals(base_value, step, steps):
    less_than = [base_value - step * i for i in range(1, steps + 1) if base_value - step * i > 0]
    more_than = [base_value + step * i for i in range(1, steps + 1)]
    return less_than[::-1] + [base_value] + more_than
