# ---- библиотеки ----#
import numpy as np
import matplotlib.pyplot as plt


# ---- необходимые параметра----#
n = 10
a = 0
b = 0.05

# ---- массы листа и жука ----#
m = 100
M = m

T = 1  # ---- параметр показывает, как быстро убывает скорость листа
# ---- при движении в вязкой жидкости в отсутствие движение жука----#

# ---- коэффициент вязкости воды ----#
alpha = 1

# ---- скорость жука ----#
w1 = np.linspace(a, b, n)
w2 = np.linspace(b, b, n)
w3 = np.linspace(b, a, n)
w = np.concatenate((w1, w2, w3))

# ---- время движения жука----#
t1 = np.linspace(0, 0.1, n)
t2 = np.linspace(1, 1, n)
t3 = np.linspace(1, 1.1, n)
t = np.concatenate((t1, t2, t3))

# ---- множитель для нахождения скорости центра масс листа ----#
argum = np.zeros(3 * n - 1)
for i in range(3 * n - 1):
    argum[i] = m / ((m + M) * T) * np.exp(-t[i] / T)

# ---- интеграл для скорости центра масс листа и его координат----#
integ = np.zeros(3 * n - 1)
for i in range(3 * n - 1):
    integ[i] = T * w[i] * np.exp(t[i] / T) - T * w[i]

# ---- изменение скорости центра масс листа до момента, ----#
# ---- когда движение жука относительно листа прекращается ----#
ans = integ * argum

# ---- момент, когда жук останавливается ----#
tau = np.max(t[:-1])

# ---- вывод графика зависимости скорости жука от времени ----#
plt.figure(figsize=(12, 2))
plt.plot(t, w)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlim([0, 11])
plt.ylim([0, b + 0.01])
plt.minorticks_on()
plt.title(r"$График\;зависимости\;\omega(t)$", fontsize=20)
plt.xlabel("t", fontsize=20)
plt.ylabel(r"$\omega$", fontsize=20)
plt.show()

# ---- скорость центра масс листа в момент остановки жука ----#
anso = ans[28]

# ---- координаты листа после момента остановки жука ----#
xt = -(m + M) / alpha * ans

# ---- координата листа в момент остановки жука ----#
xot = xt[28]

# ---- время после момента остановки жука ----#
tim = np.linspace(1.1, 11, 3 * n)

vtau = np.zeros(len(tim))
xtau = np.zeros(len(tim))

# ----скорость центра масс листа и его координаты после момента остановки жука - ---  #
for i in range(len(tim)):
    vtau[i] = anso * np.exp(-(tim[i] - tau) / T)
    xtau[i] = xot * np.exp(-(tim[i] - tau) / T)

# ---- объединение скорости центра масс листа и его координат ----#
# ---- до и после момента остановки жука ----#
X = np.hstack((xt, xtau))
V = np.hstack((ans, vtau))

# ---- объединение времени до и после момента остановки жука ----#
time = np.hstack((t[:-1], tim))

# ---- вывод графика зависимости координат листа от времени  ----#
plt.figure(figsize=(12, 2))
plt.plot(time, X)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlim([0, 11])
plt.ylim([-4, 0])
plt.minorticks_on()
plt.title(r"$График\;зависимости\;X(t)$", fontsize=20)
plt.xlabel("t", fontsize=20)
plt.ylabel("X", fontsize=20)
plt.show()

# ---- вывод графика зависимости скорости центра масс листа от времени  ----#
plt.figure(figsize=(12, 2))
plt.plot(time, V)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlim([0, 11])
plt.ylim([0, 0.018])
plt.minorticks_on()
plt.title(r"$График\;зависимости\;V(t)$", fontsize=20)
plt.xlabel("t", fontsize=20)
plt.ylabel("V", fontsize=20)
plt.show()
