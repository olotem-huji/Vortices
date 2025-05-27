import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# plt.style.use("physrev.mlpstyle")


def hyperbola_parabola(x, a, b, c, d):
    return a / (x + c) + b * x**2 + d


def hyperbola(x, a, b, c):
    return a/(x-b)**2 + c

def parabola(x, a, b, c):
    return a * (x + b)**2 + c

def get_fit(x, y, func_name="hyperbola", side="left"):
    x_data = np.array(x)
    y_data = np.array(y)

    sorted_indices = np.argsort(x_data)
    x_data = x_data[sorted_indices]
    y_data = y_data[sorted_indices]

    # Choose the fitting function
    b_lower = max(x) * 1.1 if side=="left" else min(x) * 0.9
    b_upper = b_lower * 2 if side=="left" else b_lower / 2
    p0 = [-1, (b_lower + b_upper)/2, 240]  # a, b (asymptote to right of data), c

    # Fit only to x-values to the left of the asymptote guess (optional filter)
    # mask = x < p0[1]  # keep only left branch
    # x_fit = x_data[mask]
    # y_fit = y_data[mask]

    # Fit curve
    func = hyperbola
    lower_bounds = [-np.inf, min(b_lower, b_upper), -np.inf]
    upper_bounds = [0, max(b_lower, b_upper), np.inf]
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0, bounds=(lower_bounds, upper_bounds))

    # Generate smooth x values for plotting the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 300)
    y_fit = func(x_fit, *popt)

    # Plot
    # plt.scatter(x_data, y_data, label="Data", color="red")
    formatted_params = [f"{p:.3g}" for p in popt]
    param_str = ", ".join(formatted_params)
    label = f"Fit ({func_name}), [{param_str}]"

    plt.plot(x_fit, y_fit, label=label)
    # plt.plot(x_fit, func(x_fit, 80, 200, 100, 2000))
    plt.legend()
    plt.title("Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.show()

    # Print fitted parameters
    print(f"Fitted parameters for {func_name}: {popt}")
