import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.FileWriter;
import java.io.PrintWriter;

public class Ai {
    public static void main(String[] args) {
        try {
            // Load dataset from ARFF file
            DataSource source = new DataSource("C:\\Users\\Lenovo\\Desktop\\AI\\project2\\car_evaluation.arff");
            Instances data = source.getDataSet();

            // Set the class attribute (target) as the last attribute if not already set
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Shuffle the data to ensure randomness
            Randomize randomize = new Randomize();
            randomize.setInputFormat(data);
            Instances shuffledData = Filter.useFilter(data, randomize);

            // ---------- Model M1: 70% Training, 30% Testing ----------
            System.out.println("*******  M1 (70% training and 30% test data) *******");

            int trainSizeM1 = (int) Math.round(shuffledData.numInstances() * 0.7);
            int testSizeM1 = shuffledData.numInstances() - trainSizeM1;

            Instances trainDataM1 = new Instances(shuffledData, 0, trainSizeM1);
            Instances testDataM1 = new Instances(shuffledData, trainSizeM1, testSizeM1);

            // Train J48 decision tree on training data
            J48 treeM1 = new J48();
            treeM1.buildClassifier(trainDataM1);

            // Evaluate the model on test data
            Evaluation evalM1 = new Evaluation(trainDataM1);
            evalM1.evaluateModel(treeM1, testDataM1);

            // Save decision tree in DOT format for visualization
            saveTreeAsDotFile(treeM1, "decision_tree_M1.dot");

            // Save evaluation results to a file
            saveResultsToFile("results_M1.txt", treeM1, evalM1);

            // Save class distribution for training and test sets
            saveClassDistributionToFile(trainDataM1, "M1_train_distribution.txt", "Target Class Distribution (M1 Training Data)");
            saveClassDistributionToFile(testDataM1, "M1_test_distribution.txt", "Target Class Distribution (M1 Test Data)");

            // ---------- Model M2: 50% Training, 50% Testing ----------
            System.out.println("*******  M2 (50% training and 50% test data) *******");

            int trainSizeM2 = (int) Math.round(shuffledData.numInstances() * 0.5);
            int testSizeM2 = shuffledData.numInstances() - trainSizeM2;

            Instances trainDataM2 = new Instances(shuffledData, 0, trainSizeM2);
            Instances testDataM2 = new Instances(shuffledData, trainSizeM2, testSizeM2);

            // Train another J48 model
            J48 treeM2 = new J48();
            treeM2.buildClassifier(trainDataM2);

            // Evaluate the model
            Evaluation evalM2 = new Evaluation(trainDataM2);
            evalM2.evaluateModel(treeM2, testDataM2);

            // Save results and visualizations
            saveTreeAsDotFile(treeM2, "decision_tree_M2.dot");
            saveResultsToFile("results_M2.txt", treeM2, evalM2);
            saveClassDistributionToFile(trainDataM2, "M2_train_distribution.txt", "Target Class Distribution (M2 Training Data)");
            saveClassDistributionToFile(testDataM2, "M2_test_distribution.txt", "Target Class Distribution (M2 Test Data)");

        } catch (Exception e) {
            e.printStackTrace(); // Print any error messages
        }
    }

    // Save the decision tree as a DOT graph file
    private static void saveTreeAsDotFile(J48 tree, String filePath) {
        try {
            String graph = tree.graph();
            FileWriter writer = new FileWriter(filePath);
            writer.write(graph);
            writer.close();
            System.out.println("Graph saved to: " + filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Save evaluation metrics and tree structure to a file
    private static void saveResultsToFile(String filePath, J48 tree, Evaluation eval) {
        try {
            PrintWriter writer = new PrintWriter(new FileWriter(filePath));

            writer.println("******* Decision Tree *******");
            writer.println(tree);

            writer.println("\n===== ************* =====");
            writer.println(" (Accuracy): " + eval.pctCorrect() + "%");
            writer.println("F1-Score: " + eval.weightedFMeasure());
            writer.println(eval.toMatrixString());

            writer.close();
            System.out.println("Results saved to: " + filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Save class distribution statistics (count and percentage)
    private static void saveClassDistributionToFile(Instances data, String fileName, String title) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(fileName))) {
            writer.println(title);
            writer.println("***************");

            int numInstances = data.numInstances();
            int[] classCounts = new int[data.numClasses()];

            // Count instances per class
            for (int i = 0; i < numInstances; i++) {
                int classIndex = (int) data.instance(i).classValue();
                classCounts[classIndex]++;
            }

            // Output class name, count, and percentage
            for (int i = 0; i < classCounts.length; i++) {
                String className = data.classAttribute().value(i);
                double percentage = (classCounts[i] / (double) numInstances) * 100;
                writer.printf("Class: %s, Count: %d, Percentage: %.2f%%%n", className, classCounts[i], percentage);
            }

            System.out.println("The distribution has been saved to the file: " + fileName);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
