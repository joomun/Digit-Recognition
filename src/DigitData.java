public class DigitData {
    private double[] features; // Pixel values
    private int label; // The digit label

    public DigitData(double[] features, int label) {
        this.features = features;
        this.label = label;
    }

    public double[] getFeatures() {
        return features;
    }

    public int getLabel() {
        return label;
    }
}
