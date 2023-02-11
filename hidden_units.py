import matplotlib.pyplot as plt

x = [5, 20, 50, 100, 200]
y1 = [1.8801621313531085, 1.5366914052309786, 1.2606277952584486, 1.002574759557481, 0.8383548516217763]
y2 = [1.9084412755279043, 1.6306463888000702, 1.3994728517275403, 1.2466502915323383, 1.1649715767984457]
x_labels = [5, 20, 50, 100, 200]
plt.plot(x, y1, label="Train Loss")
plt.plot(x, y2, label="Validation Loss")
plt.legend()
plt.title('Avg. Train and Validation Cross-Entropy Loss')
plt.ylabel('Average Cross-Entropy Loss')
plt.xlabel('Number of Hidden Units')
plt.xticks(ticks=x, labels=x_labels)
plt.savefig('2_1_a.png')
