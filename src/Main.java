import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Main class for executing cross-validation on different machine learning models.
 */
public class Main {

    public static void main(String[] args) {
        NearestNeighbor nn = new NearestNeighbor();
        KNearestNeighbors knn = new KNearestNeighbors(3); // Example using k=3 for KNN
        MLP mlp = new MLP(64, 64, 64); // Adjust MLP initialization as needed

        // Define file paths
        String dataSet1Path = "resources/DataSet/cw2DataSet1.csv";
        String dataSet2Path = "resources/DataSet/cw2DataSet2.csv";

        // First fold: Train on dataSet1, Test on dataSet2
        System.out.println("First fold: Training on " + dataSet1Path + " and Testing on " + dataSet2Path);
        performFold(dataSet1Path, dataSet2Path, nn, knn, mlp);
        String filename = "MLP_Biases_Fold_1.csv"; // For the first fold
     // For the second fold, you might change this to "MLP_Biases_Fold_2.csv"
        mlp.saveBiasesToCSV(filename);
        mlp = new MLP(64, 64, 64); // Reinitialize to reset the model

        // Second fold: Train on dataSet2, Test on dataSet1
        System.out.println("Second fold: Training on " + dataSet2Path + " and Testing on " + dataSet1Path);
        performFold(dataSet2Path, dataSet1Path, nn, knn, mlp);
        // After the second fold
        mlp.saveBiasesToCSV("MLP_Final_Biases.csv");
    }

    /**
     * Performs a fold of cross-validation.
     *
     * @param trainingFilePath The file path of the training data.
     * @param testingFilePath The file path of the testing data.
     * @param nn The NearestNeighbor instance.
     * @param knn The KNearestNeighbors instance.
     * @param mlp The MLP instance.
     */
    private static void performFold(String trainingFilePath, String testingFilePath, NearestNeighbor nn, KNearestNeighbors knn, MLP mlp) {
        // Load and train models on training data
        System.out.println("Training with: " + trainingFilePath);
        System.out.println("Testing with: " + testingFilePath);
        loadTrainingData(trainingFilePath, nn, knn, mlp);

        // Load test data and evaluate models
        List<DigitData> testData = loadTestData(testingFilePath);
        evaluateMLPModel(mlp, testData);
        
        System.out.println("---------------------------------------------------------------------------------------------------");
    }

    /**
     * Loads training data from a file and trains the models.
     *
     * @param filePath The file path of the training data.
     * @param nn The NearestNeighbor instance.
     * @param knn The KNearestNeighbors instance.
     * @param mlp The MLP instance.
     */
    private static void loadTrainingData(String filePath, NearestNeighbor nn, KNearestNeighbors knn, MLP mlp) {
        List<DigitData> trainingData = new ArrayList<>();
        List<int[]> targets = new ArrayList<>(); // For MLP training

        // Initialize arrays to store initial bias values for both hidden and output layers
        double[] initialHiddenBias = new double[mlp.getHiddenSize()];
        double[] initialOutputBias = new double[mlp.getOutputSize()];

        try (Scanner scanner = new Scanner(new File(filePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] columns = line.split(",");
                double[] features = new double[columns.length - 1];
                for (int i = 0; i < columns.length - 1; i++) {
                    features[i] = Double.parseDouble(columns[i]);
                }
                int label = Integer.parseInt(columns[columns.length - 1]);
                DigitData data = new DigitData(features, label);
                nn.addTrainingData(data);
                knn.addTrainingData(data);
                trainingData.add(data); // Collect training data for MLP

                // Prepare target output for MLP (assuming one-hot encoding)
                int[] targetOutput = new int[mlp.getOutputSize()]; // Assuming mlp.outputSize matches number of classes
                targetOutput[label] = 1; // Set the index of the label to 1 for one-hot encoding
                targets.add(targetOutput);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // Record initial bias values
        
        System.arraycopy(mlp.getHiddenBias(), 0, initialHiddenBias, 0, mlp.getHiddenSize());
        System.arraycopy(mlp.getOutputBias(), 0, initialOutputBias, 0, mlp.getOutputSize());

        // Convert trainingData and targets into arrays for MLP training
        double[][] inputs = new double[trainingData.size()][trainingData.get(0).getFeatures().length];
        int[][] outputs = new int[targets.size()][mlp.getOutputSize()];
        for (int i = 0; i < trainingData.size(); i++) {
            inputs[i] = trainingData.get(i).getFeatures();
            outputs[i] = targets.get(i);
        }

        // Train MLP here
        int epochs = 100; // Example epoch count
        double learningRate = 0.001; // Example learning rate
        mlp.train(inputs, outputs, epochs, learningRate);
        // Evaluate and print accuracy at the end of each epoch
        double epochAccuracy = mlp.evaluateAccuracy(inputs, targets); // Using training data for simplicity
        
        double finalTrainingAccuracy = mlp.evaluateAccuracy(inputs, outputs); // Assuming this method exists and calculates accuracy
        System.out.println("Final Training Accuracy: " + finalTrainingAccuracy + "%");


    }

    /**
     * Prints the initial and final bias values for a layer.
     *
     * @param initialBias The initial bias values.
     * @param finalBias The final bias values.
     * @param layerName The name of the layer.
     */
    private static void printBiasValues(double[] initialBias, double[] finalBias, String layerName) {
        StringBuilder biasOutput = new StringBuilder();
        for (int i = 0; i < initialBias.length; i++) {
            biasOutput.append(initialBias[i]).append("_").append(finalBias[i]);
            if (i < initialBias.length - 1) {
                biasOutput.append(", ");
            }
        }
        System.out.println(layerName + " Values: " + biasOutput.toString());
    }

    /**
     * Evaluates the MLP model's accuracy.
     *
     * @param mlp The MLP instance.
     * @param testData The test data.
     */
    private static void evaluateMLPModel(MLP mlp, List<DigitData> testData) {
        int correctPredictions = 0;
        for (DigitData data : testData) {
            int predictedLabel = mlp.predict(data.getFeatures()); // Adjust predict method to return the most probable label
            if (predictedLabel == data.getLabel()) {
                correctPredictions++;
            }
        }
        double accuracy = 100.0 * correctPredictions / testData.size();
        System.out.println("MLP Accuracy: " + accuracy + "%");
    }

    /**
     * Loads test data from a file.
     *
     * @param filePath The file path of the test data.
     * @return The loaded test data.
     */
    private static List<DigitData> loadTestData(String filePath) {
        List<DigitData> testData = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(filePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] columns = line.split(",");
                double[] features = new double[columns.length - 1];
                for (int i = 0; i < columns.length - 1; i++) {
                    features[i] = Double.parseDouble(columns[i]);
                }
                int label = Integer.parseInt(columns[columns.length - 1]);
                testData.add(new DigitData(features, label));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return testData;
    }

    /**
     * Evaluates the accuracy of a model.
     *
     * @param model The model instance.
     * @param testData The test data.
     * @param modelName The name of the model.
     */
    private static void evaluateModel(Object model, List<DigitData> testData, String modelName) {
        int correctPredictions = 0;
        for (DigitData data : testData) {
            int predictedLabel = -1;
            if (model instanceof NearestNeighbor) {
                predictedLabel = ((NearestNeighbor) model).classify(data.getFeatures());
            } else if (model instanceof KNearestNeighbors) {
                predictedLabel = ((KNearestNeighbors) model).classify(data.getFeatures());
            }

            if (predictedLabel == data.getLabel()) {
                correctPredictions++;
            }
        }
        double accuracy = 100.0 * correctPredictions / testData.size(); // Multiply by 100 to get percentage
        System.out.println(modelName + " Accuracy: " + accuracy + "%");
    }
}
