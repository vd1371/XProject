import keras
from keras.models import Sequential, load_model
import keras.losses
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l1, l2

from ._get_callbacks import get_callbacks
from ._fit_models import fit_models
import utils

class TheModel:

	def __init__(self, **params):

		for k, v in params.items():
			setattr(self, k, v)

		self.l = l1 if self.regul_type == 'l1' else l2

	def construct(self, warm_up):
		constructed = False
		if warm_up:
			try:
				self.load()
				constructed = True
			except OSError:
				self._construct_all()
		else:
			self._construct_all()

	def _construct_all(self):
		self.voters = []
		for _ in range(self.n_voters):
			self.voters.append(self._construct_one_model())

		self._log_structue()

	def _log_structue(self):

		# A summary of the model
		stringlist = []
		self.voters[0].summary(print_fn=lambda x: stringlist.append(x))
		short_model_summary = "\n".join(stringlist)
		self.log.info(short_model_summary)

	def _construct_one_model(self):

		model = Sequential()
		model.add(Dense(self.layers[0],
						input_dim = self.input_dim,
						activation = self.input_activation_func,
						kernel_regularizer=self.l(self.reg_param)))
		for ind in range(1,len(self.layers)):
			model.add(Dense(self.layers[ind],
							activation = self.hidden_activation_func,
							kernel_regularizer=self.l(self.reg_param)))
		model.add(Dense(1, activation = self.final_activation_func))
		 
		# Compile model
		model.compile(loss=self.loss_func,
						optimizer=self.optimizer,
						metrics = ['mape'])

		return model

	def fit(self):

		callbacks = get_callbacks(**self.__dict__)
		self.voters = fit_models(callbacks, **self.__dict__)


	def save(self):
		save_address = self.directory + "/" + self.dl.file_name
		for i, voter in enumerate(self.voters):
			voter.save(save_address + f"-SavedModel-Voter{i}.h5", save_format = 'h5')

	def load(self):
		self.voters = []
		model_type = 'BestModel' if self.should_checkpoint else 'SavedModel'

		self.log.info("\n\n------------\nA trained model is loaded\n------------\n\n")
		for i in range(self.n_voters):
			
			model = load_model(f"{self.directory}/{self.dl.file_name}-{model_type}-Voter{i}.h5")
			self.voters.append(model)
