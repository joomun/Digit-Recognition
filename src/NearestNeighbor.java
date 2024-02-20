import java.util.ArrayList;
import java.util.List;

public class NearestNeighbor {
    private List<DigitData> trainingData;

    public NearestNeighbor() {
        trainingData = new ArrayList<>();
    }

    public void addTrainingData(DigitData data) {
        trainingData.add(data);
    }

    public int classify(double[] testFeatures) {
        double minDistance = Double.MAX_VALUE;
        int bestLabel = -1;

        for (DigitData data : trainingData) {
            double distance = calculateEuclideanDistance(data.getFeatures(), testFeatures);
            if (distance < minDistance) {
                minDistance = distance;
                bestLabel = data.getLabel();
            }
        }

        return bestLabel;
    }

    private double calculateEuclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}
