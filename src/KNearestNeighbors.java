import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class KNearestNeighbors {
    private List<DigitData> trainingData;
    private int k;

    public KNearestNeighbors(int k) {
        this.k = k;
        this.trainingData = new ArrayList<>();
    }

    public void addTrainingData(DigitData data) {
        trainingData.add(data);
    }

    public int classify(double[] testFeatures) {
        PriorityQueue<DigitDataDistance> pq = new PriorityQueue<>(Comparator.comparingDouble(DigitDataDistance::getDistance));

        for (DigitData data : trainingData) {
            double distance = calculateEuclideanDistance(data.getFeatures(), testFeatures);
            pq.add(new DigitDataDistance(data, distance));
            if (pq.size() > k) pq.poll();
        }

        int[] votes = new int[10]; // Assuming digit labels are 0-9
        while (!pq.isEmpty()) {
            DigitDataDistance nearest = pq.poll();
            votes[nearest.getData().getLabel()]++;
        }

        int maxVotes = 0;
        int predictedLabel = -1;
        for (int i = 0; i < votes.length; i++) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i];
                predictedLabel = i;
            }
        }

        return predictedLabel;
    }

    private double calculateEuclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    private class DigitDataDistance {
        private DigitData data;
        private double distance;

        public DigitDataDistance(DigitData data, double distance) {
            this.data = data;
            this.distance = distance;
        }

        public DigitData getData() {
            return data;
        }

        public double getDistance() {
            return distance;
        }
    }
}
