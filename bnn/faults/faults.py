from bnn import bnn,util
import os


### Fault Test Class
#Uses the given network type and image set to run a series of fault tests on the network. The network is reset between runs.
#CNVFaultTest and LFCFaultTest provide class methods for easier instantiation of test with a specific classifier and dataset.
#
#Constructor Arguments:
#    classifier_cls: The classifier class type to use (e.g. bnn.CnvClassifier, bnn.LfcClassifier)
#           network: The network to test. e.g. bnn.NETWORK_CNVW1A1
#           dataset: The dataset that's being tested. (`'cifar10'`, `'streetview'`, or `'road-signs'`)
#        input_file: The file, or list of Images for non-cifar tests, containing the images to be tested.
#            labels: The list of ground truth labels for calculating accuracy.
#  control_accuracy: Optional parameter specifying the control accuracy of the network. Will be calculated if not provided.
#
#FaultTest.run_test() Arguments:
#             num_runs: The number of tests to perform
#            num_flips: The number of faults to inject per run
#            flip_word: A boolean indicating if a bit or word should be flipped
#        target_layers: An array of integers specifying which layers to target. Leave empty to target all layers.
#  weight_or_threshold: An integer specifying if weights, thresholds, or both should be targeted. Obtain these values using the static methods in the TargetType class for clarity.
#                       -1 = target weights and thresholds
#                        0 = target weights only
#                        1 = target thresholds only
class FaultTest:
	class TargetType:
		@staticmethod
		def any():
			return -1

		@staticmethod
		def weights():
			return 0

		@staticmethod
		def thresholds():
			return 1

	def __init__(self, classifier_cls, network, dataset, input_file, labels):
		self.classifier_cls = classifier_cls
		self.network = network
		self.dataset = dataset
		self.input_file = input_file
		self.labels = labels

	def run_test(self, num_runs, num_flips, flip_word=False, target_type=TargetType.any(), target_layers=[]):
		results    = [ None for i in range(num_runs) ]
		times      = [ None for i in range(num_runs) ]
		accuracies = [ None for i in range(num_runs) ]

		for i in range(num_runs):
			classifier = self.classifier_cls(self.network, self.dataset, bnn.RUNTIME_HW)

			message = "{}-{} run {} of {} (flipping {}{} {}(s) in {})".format(
				self.network,
				self.dataset,
				i+1,
				num_runs,
				num_flips,
				' weight' if target_type == 0 else ' threshold' if target_type == 1 else '',
				'word' if flip_word else 'bit',
				'any layer' if len(target_layers) == 0 else 'layer(s) {}'.format(target_layers)
			)
			print(message)

			if self.dataset == 'cifar10':
				results[i] = classifier.classify_cifars_with_faults(self.input_file, num_flips, flip_word, target_type, target_layers).tolist()
			elif self.dataset == 'mnist':
				results[i] = classifier.classify_mnists_with_faults(self.input_file, num_flips, flip_word, target_type, target_layers).tolist()
			else:
				results[i] = classifier.classify_images_with_faults(self.input_file, num_flips, flip_word, target_type, target_layers).tolist()

			times[i]      = classifier.usecPerImage
			accuracies[i] = util.calculate_accuracy(results[i], self.labels)

			print("Accuracy:", accuracies[i])
			print()

		return (results, times, accuracies)


class CNVFaultTest(FaultTest):
	def __init__(self, network, dataset, input_file, labels):
		super().__init__(bnn.CnvClassifier, network, dataset, input_file, labels)

	@classmethod
	def CIFARTest(cls, network, input_file, labels):
		return cls(network, 'cifar10', input_file, labels)

	@classmethod
	def SVHNTest(cls, network, input_file, labels):
		return cls(network, 'streetview', input_file, labels)

	@classmethod
	def GTSRBTest(cls, network, input_file, labels):
		return cls(network, 'road-signs', input_file, labels)


class LFCFaultTest(FaultTest):
	def __init__(self, network, dataset, input_file, labels):
		super().__init__(bnn.LfcClassifier, network, dataset, input_file, labels)

	@classmethod
	def MNISTTest(cls, network, input_file, labels):
		return cls(network, 'mnist', input_file, labels)





### Network Test Class
#Takes a FaultTest object and use it to run a comprehensive series of tests on different combinations of targets and SEU/MBU
#
#Constructor Arguments:
#     fault_test: A FaultTest (or derived class) object
#  output_folder: The folder to store the test results in. Results will be stored in subfolders organized by network and dataset.
#
#NetworkTest.test_network() Arguments:
#       num_runs: The number of runs in each test
#    flip_counts: A list of fault counts to inject each run. One test will be run for each.
#     test_types: A list of TestType objects which specify the location and type of fault for each test.
#  target_layers: An array of integers specifying which layers to target. Leave empty to target all layers.
class NetworkTest:
	class TestType:
		def __init__(self, target_type, flip_word):
			self.target_type = target_type
			self.flip_word = flip_word
			self.__build_name()

		def __build_name(self):
			self.name = ''

			if self.target_type == FaultTest.TargetType.any():
				self.name += 'any'
			elif self.target_type == FaultTest.TargetType.weights():
				self.name += 'weight'
			else:
				self.name += 'threshold'

			self.name += ' '

			if self.flip_word:
				self.name += 'word'
			else:
				self.name += 'bit'

		@classmethod
		def any_bit(cls):
			return cls(FaultTest.TargetType.any(), False)

		@classmethod
		def any_word(cls):
			return cls(FaultTest.TargetType.any(), True)

		@classmethod
		def weight_bit(cls):
			return cls(FaultTest.TargetType.weights(), False)

		@classmethod
		def weight_word(cls):
			return cls(FaultTest.TargetType.weights(), True)

		@classmethod
		def threshold_bit(cls):
			return cls(FaultTest.TargetType.thresholds(), False)

		@classmethod
		def threshold_word(cls):
			return cls(FaultTest.TargetType.thresholds(), True)


	def __init__(self, fault_test, output_folder):
		self.fault_test = fault_test
		self.output_folder = output_folder + '/' + self.fault_test.network + '/' + self.fault_test.dataset + '/'
		self.control = None

	def __run_control(self):
		print("Running", self.fault_test.network + "-" + self.fault_test.dataset, "control test")
		_, _, accuracy = self.fault_test.run_test(num_runs=1, num_flips=0);
		self.control = accuracy[0]

	# Output dict contains the combined raw results from a test. Used as input to calculate_stats
	def __build_output_dict(self, name, num_runs, num_flips, layers, accuracies):
		out = {}
		out["network"] = self.fault_test.network
		out["dataset"] = self.fault_test.dataset
		out["run count"] = num_runs
		out["flips"] = num_flips
		out["control"] = self.control
		out["layers"] = layers
		out["results"] = {}
		out["results"][name] = accuracies
		return out

	# Calculates various stats given the results of a test (formatted with __build_output_dict)
	def __calculate_stats(self, output_dict):
		out = output_dict.copy()
		for key, value in output_dict["results"].items():
			out["results"][key] = {}
			out["results"][key]["runs"] = {}
			out["results"][key]["runs"]["all"] = value
			out["results"][key]["runs"]["effective"] = list(filter(lambda x: x != out["control"], value))
			out["results"][key]["effective count"] = len(out["results"][key]["runs"]["effective"])
			out["results"][key]["min accuracy"] = min(value)
			out["results"][key]["max accuracy"] = max(value)
			if out["results"][key]["effective count"] != 0:
				out["results"][key]["avg accuracy"] = sum(value) / len(value)
				out["results"][key]["avg effective accuracy"] = sum(out["results"][key]["runs"]["effective"]) / out["results"][key]["effective count"]
				out["results"][key]["accuracy delta"] = out["control"] - out["results"][key]["avg accuracy"]
				out["results"][key]["effective accuracy delta"] = out["control"] - out["results"][key]["avg effective accuracy"]
			else:
				out["results"][key]["avg accuracy"] = out["control"]
		return out


	def __run_tests(self, folder, num_runs, num_flips, test_types, target_layers):
		output_dicts = []

		for test in test_types:
			_, _, accuracies = self.fault_test.run_test(num_runs, num_flips, test.flip_word, test.target_type, target_layers)
			output = self.__build_output_dict(test.name, num_runs, num_flips, target_layers, accuracies)
			output_dicts.append(output)
			util.write_dict_to_file(folder + '/temp/' + self.fault_test.network + '_results_' + test.name.replace(' ', '-') + '.json', output)

		# Calculate stats
		stats = util.dict_of_dicts_merge(*output_dicts)
		stats = self.__calculate_stats(stats)
		return stats


	def test_network(self, num_runs, flip_counts, test_types, target_layers=[]):
		# Calculate the control accuracy
		if self.control is None:
			self.__run_control()

		for num_flips in flip_counts:
			folder = self.output_folder + '/' + str(num_flips) + 'flips/'

			# Run all tests for this flip count
			stats = self.__run_tests(folder, num_runs, num_flips, test_types, target_layers)

			# Build file name
			filename = folder + '/' + self.fault_test.network + '_' + self.fault_test.dataset
			if len(target_layers) > 0:
				filename = filename + "_stats_layer{}.json".format(target_layers)
			else:
				filename = filename + "_stats.json"

			# Write results to the output file
			util.write_dict_to_file(filename, stats)


	def comprehensive_test(self, num_runs, flip_counts, target_layers=[]):
		test_types = [
			NetworkTest.TestType.any_bit(), NetworkTest.TestType.any_word(),
			NetworkTest.TestType.weight_bit(), NetworkTest.TestType.weight_word(),
			NetworkTest.TestType.threshold_bit(), NetworkTest.TestType.threshold_word()
		]

		self.test_network(num_runs, flip_counts, test_types, target_layers)