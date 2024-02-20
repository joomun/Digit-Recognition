import java.util.List;
import java.util.Random;

public class MLP {
    private double[][] hiddenWeights; // Hidden layer weights
    private double[] hiddenBias; // Hidden layer biases
    private double[][] outputWeights; // Output layer weights
    private double[] outputBias; // Output layer biases
    private int inputSize; // Number of inputs
    private int hiddenSize; // Number of hidden neurons
    private int outputSize; // Number of outputs
    private Random rand = new Random();

    public MLP(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.setOutputSize(outputSize);
        
        hiddenWeights = new double[hiddenSize][inputSize];
        hiddenBias = new double[hiddenSize];
        outputWeights = new double[outputSize][hiddenSize];
        outputBias = new double[outputSize];

        initializeWeights(hiddenWeights);
        initializeWeights(outputWeights);
    }

    private void initializeWeights(double[][] weights) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = rand.nextDouble() * 2 - 1; // Initialize to [-1, 1]
            }
        }
    }

    public int predict(double[] input) {
        double[] hiddenOutputs = forwardPass(input, hiddenWeights, hiddenBias);
        double[] output = forwardPass(hiddenOutputs, outputWeights, outputBias);
        return maxIndex(output);
    }

    private double[] forwardPass(double[] input, double[][] weights, double[] bias) {
        double[] layerOutput = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < input.length; j++) {
                layerOutput[i] += weights[i][j] * input[j];
            }
            layerOutput[i] += bias[i];
            layerOutput[i] = sigmoid(layerOutput[i]);
        }
        return layerOutput;
    }

    private int maxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double derivativeSigmoid(double x) {
        return x * (1 - x);
    }

    public void train(double[][] inputs, int[][] targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int sample = 0; sample < inputs.length; sample++) {
                // Forward pass
                double[] hiddenOutputs = forwardPass(inputs[sample], hiddenWeights, hiddenBias);
                double[] output = forwardPass(hiddenOutputs, outputWeights, outputBias);

                // Calculate output layer error
                double[] outputErrors = new double[this.getOutputSize()];
                for (int i = 0; i < getOutputSize(); i++) {
                    outputErrors[i] = (targets[sample][i] - output[i]) * derivativeSigmoid(output[i]);
                }

                // Calculate hidden layer error
                double[] hiddenErrors = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    hiddenErrors[i] = 0;
                    for (int j = 0; j < getOutputSize(); j++) {
                        hiddenErrors[i] += outputErrors[j] * outputWeights[j][i];
                    }
                    hiddenErrors[i] *= derivativeSigmoid(hiddenOutputs[i]);
                }

                // Update output weights
                for (int i = 0; i < getOutputSize(); i++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        outputWeights[i][j] += learningRate * outputErrors[i] * hiddenOutputs[j];
                    }
                    outputBias[i] += learningRate * outputErrors[i];
                }

                // Update hidden weights
                for (int i = 0; i < hiddenSize; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        hiddenWeights[i][j] += learningRate * hiddenErrors[i] * inputs[sample][j];
                    }
                    hiddenBias[i] += learningRate * hiddenErrors[i];
                }
            }
            
            // After each epoch, evaluate the model's accuracy on the entire training set
            double accuracy = this.evaluateAccuracy(inputs, targets);
            System.out.println("Epoch " + (epoch + 1) + " Accuracy: " + accuracy + "%");
        }
    }


    public double evaluateAccuracy(double[][] inputs, int[][] targets) {
        int correctPredictions = 0;
        for (int i = 0; i < inputs.length; i++) {
            int predictedLabel = predict(inputs[i]);
            
            int actualLabel = -1;
            for (int j = 0; j < targets[i].length; j++) {
                if (targets[i][j] == 1) {
                    actualLabel = j;
                    break;
                }
            }
            if (predictedLabel == actualLabel) {
                correctPredictions++;
            }
        }
        return 100.0 * correctPredictions / inputs.length;
    }


	public double evaluateAccuracy(double[][] inputs, List<int[]> targets) {
	    int correctPredictions = 0;
	    for (int i = 0; i < inputs.length; i++) {
	        int predictedLabel = predict(inputs[i]);
	        
	        int actualLabel = -1;
	        for (int j = 0; j < targets.get(i).length; j++) {
	            if (targets.get(i)[j] == 1) {
	                actualLabel = j;
	                break;
	            }
	        }
	        if (predictedLabel == actualLabel) {
	            correctPredictions++;
	        }
	    }
	    return 100.0 * correctPredictions / inputs.length;
	}
	public int getOutputSize() {
		return outputSize;
	}

	public void setOutputSize(int outputSize) {
		this.outputSize = outputSize;
	}


}
