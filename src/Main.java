import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
	    NearestNeighbor nn = new NearestNeighbor();
	    KNearestNeighbors knn = new KNearestNeighbors(3); // Example using k=3 for KNN
	    MLP mlp = new MLP(64, 128, 10); // Adjust MLP initialization as needed

	    // Define file paths
	    String dataSet1Path = "resources/DataSet/cw2DataSet1.csv";
	    String dataSet2Path = "resources/DataSet/cw2DataSet2.csv";

	    // First fold: Train on dataSet1, Test on dataSet2
	    System.out.println("First fold: Training on " + dataSet1Path + " and Testing on " + dataSet2Path);
	    performFold(dataSet1Path, dataSet2Path, nn, knn, mlp);

	    // Reset models for second fold (if necessary)
	    nn = new NearestNeighbor();
	    knn = new KNearestNeighbors(3);
	    mlp = new MLP(64, 128, 10); // Reinitialize to reset the model

	    // Second fold: Train on dataSet2, Test on dataSet1
	    System.out.println("Second fold: Training on " + dataSet2Path + " and Testing on " + dataSet1Path);
	    performFold(dataSet2Path, dataSet1Path, nn, knn, mlp);
	}

	private static void performFold(String trainingFilePath, String testingFilePath, NearestNeighbor nn, KNearestNeighbors knn, MLP mlp) {
	    // Load and train models on training data
	    loadTrainingData(trainingFilePath, nn, knn, mlp);

	    // Load test data and evaluate models
	    List<DigitData> testData = loadTestData(testingFilePath);
	    evaluateModel(nn, testData, "Nearest Neighbor");
	    evaluateModel(knn, testData, "K-Nearest Neighbors");
	    evaluateMLPModel(mlp, testData);
	}

    private static void loadTrainingData(String filePath, NearestNeighbor nn, KNearestNeighbors knn, MLP mlp) {
        List<DigitData> trainingData = new ArrayList<>();
        List<int[]> targets = new ArrayList<>(); // For MLP training
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

        // Convert trainingData and targets into arrays for MLP training
        double[][] inputs = new double[trainingData.size()][trainingData.get(0).getFeatures().length];
        int[][] outputs = new int[targets.size()][mlp.getOutputSize()];
        for (int i = 0; i < trainingData.size(); i++) {
            inputs[i] = trainingData.get(i).getFeatures();
            outputs[i] = targets.get(i);
        }

        // Train MLP here
        int epochs = 400; // Example epoch count
        double learningRate = 0.01; // Example learning rate
        mlp.train(inputs, outputs, epochs, learningRate);
        // Evaluate and print accuracy at the end of each epoch
        double epochAccuracy = mlp.evaluateAccuracy(inputs, targets);// Using training data for simplicity
        System.out.println("Epoch " + (epochs + 1) + " Accuracy: " + epochAccuracy + "%");

    }


    

	// Evaluate MLP model separately due to its different output format
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

    // Updated evaluateModel to accept a model name for differentiating output
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

