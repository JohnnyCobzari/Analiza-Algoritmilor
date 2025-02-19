import time
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(20000)
matplotlib.use('TkAgg')

# Define the Fibonacci function
def nth_fibonacci(n):
    if n <= 1:
        return n
    return nth_fibonacci(n - 1) + nth_fibonacci(n - 2)


def nth_fibonacci2(n, memo={}):
    if n <= 1:
        return n

    # If value is already computed, return it
    if n in memo:
        return memo[n]

    # Recursively compute and store in memo
    memo[n] = nth_fibonacci2(n - 1, memo) + nth_fibonacci2(n - 2, memo)

    return memo[n]


def nth_fibonacci3(n):
    # Handle the edge cases
    if n <= 1:
        return n

    # Create a list to store Fibonacci numbers
    dp = [0] * (n + 1)

    # Initialize the first two Fibonacci numbers
    dp[0] = 0
    dp[1] = 1

    # Fill the list iteratively
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    # Return the nth Fibonacci number
    return dp[n]


def multiply(mat1, mat2):
    x = mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[1][0]
    y = mat1[0][0] * mat2[0][1] + mat1[0][1] * mat2[1][1]
    z = mat1[1][0] * mat2[0][0] + mat1[1][1] * mat2[1][0]
    w = mat1[1][0] * mat2[0][1] + mat1[1][1] * mat2[1][1]
    mat1[0][0], mat1[0][1] = x, y
    mat1[1][0], mat1[1][1] = z, w
def matrix_power(mat1, n):
    if n == 0 or n == 1:
        return
    mat2 = [[1, 1], [1, 0]]
    matrix_power(mat1, n // 2)
    multiply(mat1, mat1)
    if n % 2 != 0:
        multiply(mat1, mat2)
def nth_fibonacci4(n):
    if n <= 1:
        return n
    mat1 = [[1, 1], [1, 0]]
    matrix_power(mat1, n - 1)
    return mat1[0][0]


def nth_fibonacci5(n):
    if n <= 1:
        return n
    curr = 0
    prev1 = 1
    prev2 = 0
    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    return curr


from decimal import Decimal, getcontext

def nth_fibonacci6(n):

    if n <= 1:
        return n

    getcontext().prec = n // 2
    sqrt5 = Decimal(5).sqrt()
    phi = (Decimal(1) + sqrt5) / Decimal(2)
    psi = (Decimal(1) - sqrt5) / Decimal(2)
    fib_n = (phi**n - psi**n) / sqrt5

    return int(fib_n)





inputs = [5,9,10,12,15,20,24,26,30,35]# Increasing values to observe execution time growth
inputs2= [334, 369, 379, 939, 2006, 2213,2835, 3021, 3256, 4109, 4401,  5628, 6013, 6320, 7251, 7479, 8317, 8591, 9382, 9624, 9880,  10756, 11022, 11999, 13012,15000,21000,25000,30000,35000,40000,50000]
# Measure execution time for each input
execution_times2 = []
execution_times3 = []
execution_times4 = []
execution_times5 = []
execution_times6 = []
for n in inputs2:
    start_time = time.time()
    nth_fibonacci2(n)
    end_time = time.time()
    execution_times2.append(end_time - start_time)

for n in inputs2:
    start_time = time.time()
    nth_fibonacci3(n)
    end_time = time.time()
    execution_times3.append(end_time - start_time)


for n in inputs2:
    start_time = time.time()
    nth_fibonacci4(n)
    end_time = time.time()
    execution_times4.append(end_time - start_time)

for n in inputs2:
    start_time = time.time()
    nth_fibonacci5(n)
    end_time = time.time()
    execution_times5.append(end_time - start_time)

for n in inputs2:
    start_time = time.time()
    nth_fibonacci6(n)
    end_time = time.time()
    execution_times6.append(end_time - start_time)
# # Create a DataFrame
# df = pd.DataFrame({"Input (n)": inputs2, "Execution Time (s)": execution_times2})
#
# # Display the table
# print(df)

plt.figure(figsize=(10, 6))
plt.plot(inputs2, execution_times2, marker="o", linestyle="-", label="Memoization (O(n))")
plt.plot(inputs2, execution_times3, marker="s", linestyle="-", label="Bottom-Up (O(n))")
plt.plot(inputs2, execution_times4, marker="^", linestyle="-", label="Matrix Exponentiation (O(log n))")
plt.plot(inputs2, execution_times5, marker="d", linestyle="-", label="Space-Optimized (O(n))")
plt.plot(inputs2, execution_times6, marker="x", linestyle="-", label="Binet's Formula (O(1))")

# Labels and title
plt.xlabel("Input (n)")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time of Fibonacci Algorithms")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
