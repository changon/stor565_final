import matplotlib.pyplot as plt
xs =[0.001, 0.01, 0.1, 1, 10, 100, 1000]
ys =[1.0]*len(xs)-[0.863965350088067, 0.863965350088067, 0.8639838566103482, 0.8639283267630553, 0.863410031054239, 0.8632619548882744, 0.8632619548882744]
plt.plot(xs, ys, "-o")
for x, y in zip(xs, ys):
    plt.text(x, y, str(x), color="red", fontsize=12)
ax.set_xlabel('Hyperparameters')
ax.set_ylabel('Classification Error')
plt.show()
