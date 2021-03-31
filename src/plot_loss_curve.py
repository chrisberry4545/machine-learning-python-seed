from matplotlib import pyplot

def plot_loss_curve(epochs, rmse):
  pyplot.figure()
  pyplot.xlabel("Epoch")
  pyplot.ylabel("Root Mean Squared Error")

  pyplot.plot(epochs, rmse, label="Loss")
  pyplot.legend()
  pyplot.ylim([rmse.min()*0.94, rmse.max()* 1.05])
  pyplot.show()
