package org.deeplearning4j;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;

import java.util.ArrayList;
import java.util.List;

public class dl4jGAN {
    private static final Logger log = LoggerFactory.getLogger(dl4jGAN.class);
    private int batchSizePerWorker = 50;
    private int numEpochs = 5;
    private int numLinesToSkip = 0;
    private String delimiter = ",";
    private int labelIndex = 784;
    private int numClasses = 10;

    public static void main(String[] args) throws Exception {
        new dl4jGAN().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("DL4J Spark MLP Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        RecordReader recordReaderTrain = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTrain.initialize(new FileSplit(new ClassPathResource("mnist_train.txt").getFile()));
        RecordReader recordReaderTest = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTest.initialize(new FileSplit(new ClassPathResource("mnist_test.txt").getFile()));

        DataSetIterator iterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSizePerWorker, labelIndex, numClasses);
        DataSetIterator iterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSizePerWorker, labelIndex, numClasses);

        List<DataSet> trainDataList = new ArrayList<>();
        List<DataSet> testDataList = new ArrayList<>();

        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }

        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);

        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(new ClassPathResource("mlp_fnctl.h5").getFile().getPath(),true);

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
                                                                .averagingFrequency(5)
                                                                .workerPrefetchNumBatches(2)
                                                                .batchSizePerWorker(batchSizePerWorker)
                                                                .build();

        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, model.getConfiguration(), tm);

        for (int i = 0; i < numEpochs; i++) {
            sparkNet.fit(trainData);
            log.info("Completed Epoch {}", i);
        }

        Evaluation evaluation = sparkNet.doEvaluation(testData, 64, new Evaluation(10))[0];
        log.info(evaluation.stats());
        tm.deleteTempFiles(sc);
    }
}
