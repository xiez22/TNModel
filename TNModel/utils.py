from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = {'batch':[], 'epoch':[]}
		self.accuracy = {'batch':[], 'epoch':[]}
 
	def on_batch_end(self, batch, logs={}):
		self.losses['batch'].append(logs.get('loss'))
		self.accuracy['batch'].append(logs.get('accuracy'))
 
	def on_epoch_end(self, batch, logs={}):
		self.losses['epoch'].append(logs.get('loss'))
		self.accuracy['epoch'].append(logs.get('accuracy'))
 
	def loss_plot(self, loss_type):
		iters = range(len(self.losses[loss_type]))
		sns.set()
		# make a figure
		fig = plt.figure(figsize=(8,4))
		# subplot loss
		ax1 = fig.add_subplot(121)
		ax1.plot(iters, self.losses[loss_type], 'g', label='train_loss')
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Loss')
		ax1.set_title('Loss on Training Data')
		ax1.legend()
		# subplot acc
		ax2 = fig.add_subplot(122)
		ax2.plot(iters, self.accuracy[loss_type], 'r', label='train_acc')
		ax2.set_xlabel('Epochs')
		ax2.set_ylabel('Accuracy')
		ax2.set_title('Accuracy on Training Data')
		ax2.legend()
		plt.tight_layout()
		plt.show()
