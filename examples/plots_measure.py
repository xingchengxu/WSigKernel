import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file_path = '.\\results\\signature_kernel_result.xlsx'

# ================================ number of sample paths n varying================================
data = pd.read_excel(file_path, sheet_name='n-vary-BM', engine='openpyxl')

print(data)

dim = 'n'
labels = ['5', '10', '15', '20', '25']

i = 0
titles = ['Correlation', 'Loss', 'KL Divergence', 'Correlation Ratio', 'Loss Ratio', 'JS Divergence']
for col in ['corr1', 'loss1', 'kl_div', 'corr_ratio', 'loss_ratio', 'js_div']:
    i += 1
    plt.subplot(2, 3, i)
    x = data[[dim, col]].dropna()

    box_data = [x[(x[dim] == 5)][col].values, x[(x[dim] == 10)][col].values, x[(x[dim] == 15)][col].values,
                x[(x[dim] == 20)][col].values, x[(x[dim] == 25)][col].values]
    plt.boxplot(box_data, labels=labels, notch=False, patch_artist=False, showfliers=False)
    plt.title(titles[i-1])

# plt.show()
plt.close()

# ================================ times steps m varying================================

data = pd.read_excel(file_path, sheet_name='m-vary-BM', engine='openpyxl')

print(data)

dim = 'm'
labels = ['5', '10', '15', '20', '25']

i = 0
titles = ['Correlation', 'Loss', 'KL Divergence', 'Correlation Ratio', 'Loss Ratio', 'JS Divergence']
for col in ['corr1', 'loss1', 'kl_div', 'corr_ratio', 'loss_ratio', 'js_div']:
    i += 1
    plt.subplot(2, 3, i)
    x = data[[dim, col]].dropna()

    box_data = [x[(x[dim] == 5)][col].values, x[(x[dim] == 10)][col].values, x[(x[dim] == 15)][col].values,
                x[(x[dim] == 20)][col].values, x[(x[dim] == 25)][col].values]
    plt.boxplot(box_data, labels=labels, notch=False, patch_artist=False, showfliers=False)
    plt.title(titles[i-1])

# plt.show()
plt.close()

# ================================ dimension d varying================================
# data = pd.read_excel(file_path, sheet_name='d-vary-BM', engine='openpyxl')
data = pd.read_excel(file_path, sheet_name='d-vary-BM-org', engine='openpyxl')

print(data)

dim = 'd'
labels = ['2', '3', '4', '5', '6']

i = 0
# titles = ['(a) Correlation', '(b) Loss']
ylabels = ['the alignment', 'the MMD distance']
for col in ['corr1', 'loss1']:
    i += 1
    plt.subplot(1, 2, i)
    x = data[[dim, col]].dropna()

    box_data = [x[(x[dim] == 2)][col].values, x[(x[dim] == 3)][col].values, x[(x[dim] == 4)][col].values,
                x[(x[dim] == 5)][col].values, x[(x[dim] == 6)][col].values]
    plt.boxplot(box_data, labels=labels, notch=False, patch_artist=False, showfliers=False)
    # plt.title(titles[i-1])
    plt.xlabel('the dimension of BM')
    plt.ylabel(ylabels[i-1])

plt.show()
plt.close()


# ================================ BM+epsilon*sin: epsilon varying ================================
data = pd.read_excel(file_path, sheet_name='e-vary-BMsin-v2-org', engine='openpyxl')

print(data)

dim = 'epsilon'

i = 0
titles = ['(a) Correlation', '(b) Correlation Ratio', '(c) Loss', '(d) Loss Ratio']
for col in ['corr1', 'corr_ratio', 'loss1', 'loss_ratio']:
    i += 1
    plt.subplot(2, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, c='k')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='darkorange', alpha=0.2)

    plt.title(titles[i-1])

# plt.show()
plt.close()

# ================================ BM+epsilon*sin: freq+epsilon varying ================================
data = pd.read_excel(file_path, sheet_name='e-vary-BMsin-v2-org', engine='openpyxl')

dim = 'epsilon'

i = 0
titles = ['(a) Correlation', '(b) Loss', 'KL Divergence', 'Correlation Ratio', 'Loss Ratio', 'JS Divergence']
for col in ['corr1', 'loss1']:
    i += 1
    plt.subplot(1, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, c='k', label=r'$\nu$=2')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='darkorange', alpha=0.2)

    plt.title(titles[i-1])

data = pd.read_excel(file_path, sheet_name='e-vary-BMsin-v3-org', engine='openpyxl')

dim = 'epsilon'

i = 0
titles = ['(a) Correlation', '(b) Loss', 'KL Divergence', 'Correlation Ratio', 'Loss Ratio', 'JS Divergence']
for col in ['corr1', 'loss1']:
    i += 1
    plt.subplot(1, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, linestyle='--', c='k', label=r'$\nu$=3')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='orange', alpha=0.2)

    plt.title(titles[i-1])

    plt.legend()
# plt.show()
plt.close()


# ================================ BM+epsilon*sin: freq+epsilon varying ================================
data = pd.read_excel(file_path, sheet_name='e-vary-BMsin-v2-org', engine='openpyxl')

dim = 'epsilon'
i = 0
titles = ['(a) '+r'$\nu=2$', '(b) '+r'$\nu=2$',
          '(c) '+r'$\nu=3$', '(d) '+r'$\nu=3$']
for col in ['corr1', 'loss1']:
    i += 1
    plt.subplot(2, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, c='k', label='optimal meas.')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='darkorange', alpha=0.2)

    plt.title(titles[i-1])
    plt.legend()

i = 0
for col in ['corr2', 'loss2']:
    i += 1
    plt.subplot(2, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, linestyle='--', c='k', label='empirical meas.')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='blue', alpha=0.2)

    plt.title(titles[i-1])

    plt.legend()

data = pd.read_excel(file_path, sheet_name='e-vary-BMsin-v3-org', engine='openpyxl')

dim = 'epsilon'
i = 2
titles = ['(a) '+r'$\nu=2$', '(b) '+r'$\nu=2$',
          '(c) '+r'$\nu=3$', '(d) '+r'$\nu=3$']
for col in ['corr1', 'loss1']:
    i += 1
    plt.subplot(2, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, c='k', label='optimal meas.')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='darkorange', alpha=0.2)

    plt.title(titles[i-1])
    plt.legend()

i = 2
for col in ['corr2', 'loss2']:
    i += 1
    plt.subplot(2, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, linestyle='--', c='k', label='empirical meas.')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='blue', alpha=0.2)

    plt.title(titles[i-1])

    plt.legend()


plt.show()
plt.close()


# ================================ BM+epsilon*Davis: epsilon varying ================================
data = pd.read_excel(file_path, sheet_name='e-vary-BM-DM-org', engine='openpyxl')

print(data)

dim = 'epsilon'

i = 0
titles = ['(a) Correlation', '(b) Correlation Ratio', '(c) Loss', '(d) Loss Ratio']
for col in ['corr1', 'corr_ratio', 'loss1', 'loss_ratio']:
    i += 1
    plt.subplot(2, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, c='k')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='darkorange', alpha=0.2)

    plt.title(titles[i-1])

# plt.show()
plt.close()


# ================================ BM+epsilon*Davis: epsilon varying ================================
# data = pd.read_excel(file_path, sheet_name='e-vary-BM-DM', engine='openpyxl')
data = pd.read_excel(file_path, sheet_name='e-vary-BM-DM-org', engine='openpyxl')

print(data)

dim = 'epsilon'

i = 0
# titles = ['(a) Correlation', '(b) Loss']
xlabels = ["value of "+r"$\epsilon$"]
ylabels = ["the alignment", "the dis-similarity", "the alignment", "the dis-similarity"]
for col in ['corr1', 'loss1']:
    i += 1
    plt.subplot(1, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, c='k', label='optimal meas.')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='darkorange', alpha=0.2)

    # plt.title(titles[i-1])
    plt.xlabel("value of "+r"$\epsilon$")
    plt.ylabel(ylabels[i - 1])
    plt.legend()

i = 0
for col in ['corr2', 'loss2']:
    i += 1
    plt.subplot(1, 2, i)
    x = data[[dim, col]].dropna()

    quant_50 = x.groupby(by=dim).quantile(0.5)
    quant_25 = x.groupby(by=dim).quantile(0.25)
    quant_75 = x.groupby(by=dim).quantile(0.75)
    plt.plot(quant_50, linestyle='--', c='k', label='empirical meas.')
    print(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0])
    plt.fill_between(quant_25.index, quant_25.values.reshape(1, -1)[0], quant_75.values.reshape(1, -1)[0],
                     color='blue', alpha=0.2)

    # plt.title(titles[i-1])
    plt.xlabel("value of " + r"$\epsilon$")
    plt.ylabel(ylabels[i - 1])
    plt.legend()
plt.show()
plt.close()
