import java.util.List;
import java.util.Random;
import java.io.FileWriter;
import java.io.IOException;

public class MLP {
    private double[][] hiddenWeights; // Hidden layer weights
    private double[] hiddenBias; // Hidden layer biases
    private double[][] outputWeights; // Output layer weights
    private double[] outputBias; // Output layer biases
    private double[] initialHiddenBias; // Initial hidden biases
    private double[] initialOutputBias; // Initial output biases
    private int inputSize; // Number of inputs
    private int hiddenSize; // Number of hidden neurons
    private int outputSize; // Number of outputs
    private Random rand = new Random();

    private double dropoutProbability;

    public MLP(int inputSize, int hiddenSize, int outputSize, double dropoutProbability) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.setOutputSize(outputSize);
        this.dropoutProbability = dropoutProbability;

        hiddenWeights = new double[hiddenSize][inputSize];
        hiddenBias = new double[hiddenSize];
        outputWeights = new double[outputSize][hiddenSize];
        outputBias = new double[outputSize];
        initialHiddenBias = new double[hiddenSize];
        initialOutputBias = new double[outputSize];

        initializeWeightsXavier(hiddenWeights, inputSize, hiddenSize);
        initializeWeightsXavier(outputWeights, hiddenSize, outputSize);
        recordInitialBiases(); // Record initial bias values
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    // Getter method for hiddenBias
    public double[] getHiddenBias() {
        return hiddenBias;
    }

    // Getter method for outputBias
    public double[] getOutputBias() {
        return outputBias;
    }

    public MLP(int inputSize, int hiddenSize, int outputSize) {
        this(inputSize, hiddenSize, outputSize, 0); // Default dropout probability is 0
    }

    private void initializeWeightsXavier(double[][] weights, int inputNeurons, int outputNeurons) {
        double xavierRange = Math.sqrt(6.0 / (inputNeurons + outputNeurons));
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = rand.nextDouble() * 2 * xavierRange - xavierRange; // Initialize within Xavier range
            }
        }
    }

    private void recordInitialBiases() {
        System.arraycopy(hiddenBias, 0, initialHiddenBias, 0, hiddenBias.length);
        System.arraycopy(outputBias, 0, initialOutputBias, 0, outputBias.length);
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

    public void printWeightsAndBiases() {
        System.out.println("Hidden Layer Weights:");
        printMatrix(hiddenWeights);
        System.out.println("Hidden Layer Biases:");
        printArray(initialHiddenBias); // Print initial hidden biases
        System.out.println("Final Hidden Layer Biases:");
        printArray(hiddenBias); // Print final hidden biases

        System.out.println("Output Layer Weights:");
        printMatrix(outputWeights);
        System.out.println("Output Layer Biases:");
        printArray(initialOutputBias); // Print initial output biases
        System.out.println("Final Output Layer Biases:");
        printArray(outputBias); // Print final output biases
    }

    private void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            for (double value : row) {
                System.out.printf("%.4f ", value);
            }
            System.out.println();
        }
    }

    private void printArray(double[] array) {
        for (double value : array) {
            System.out.printf("%.4f ", value);
        }
        System.out.println();
    }

    public void train(double[][] inputs, int[][] targets, int epochs, double initialLearningRate) {
        double learningRate = initialLearningRate;
        int[] decayEpochs = {50, 100, 150}; // Define epochs at which to decay the learning rate
        double decayFactor = 0.5; // Factor by which to decay the learning rate

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Check if current epoch is in decay epochs
            if (contains(decayEpochs, epoch)) {
                learningRate *= decayFactor; // Decay the learning rate
            }

            for (int sample = 0; sample < inputs.length; sample++) {
                // Forward pass
                double[] hiddenOutputs = forwardPass(inputs[sample], hiddenWeights, hiddenBias);
                // Apply dropout to the hidden layer
                dropout(hiddenOutputs);

                double[] output = forwardPass(hiddenOutputs, outputWeights, outputBias);

                // Calculate output layer error
                double[] outputErrors = new double[getOutputSize()];
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
            
        }
    }

    // Helper method to check if an array contains a certain value
    private boolean contains(int[] array, int value) {
        for (int num : array) {
            if (num == value) {
                return true;
            }
        }
        return false;
    }
    public void saveBiasesToCSV(String filename) {
        try (FileWriter csvWriter = new FileWriter(filename)) {
            // Write the header
            csvWriter.append("Hidden Layer Initial Bias,Hidden Layer Final Bias,Output Layer Initial Bias,Output Layer Final Bias\n");

            // Assuming the hidden and output layers have the same number of neurons
            for (int i = 0; i < hiddenBias.length; i++) {
                csvWriter.append(String.format("%.4f,%.4f,%.4f,%.4f\n",
                    initialHiddenBias[i], hiddenBias[i],
                    initialOutputBias[i], outputBias[i]));
            }
        } catch (IOException e) {
            e.printStackTrace();
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

    // Apply dropout to the hidden layer
    private void dropout(double[] array) {
        for (int i = 0; i < array.length; i++) {
            if (rand.nextDouble() < dropoutProbability) {
                array[i] = 0; // Drop the neuron
            }
        }
    }
}
