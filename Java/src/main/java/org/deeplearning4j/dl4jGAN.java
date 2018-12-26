package org.deeplearning4j;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.*;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;

import org.nd4j.linalg.activations.*;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;

import java.util.*;

public class dl4jGAN {
    private static final Logger log = LoggerFactory.getLogger(dl4jGAN.class);

    private static final int batchSizePerWorker = 50;
    private static final int numEpochs = 1;
    private static final int numLinesToSkip = 0;
    private static final int labelIndex = 784;
    private static final int numClasses = 10;

    private static final double learning_rate = 0.0015;
    private static final double frozen_learning_rate = 0.0;

    private static final String delimiter = ",";

    public static void main(String[] args) throws Exception {
        new dl4jGAN().GAN(args);
    }

    private void GAN(String[] args) throws Exception {
        for (int i = 0; i < args.length; i++) {
            System.out.println(args[i]);
        }

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("DL4J Spark Generative Adversarial Network (GAN)");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        RecordReader recordReaderTrain = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTrain.initialize(new FileSplit(new ClassPathResource("mnist_train.txt").getFile()));
        DataSetIterator iterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSizePerWorker, labelIndex, numClasses);
        List<DataSet> trainDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }
        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);

        RecordReader recordReaderTest = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTest.initialize(new FileSplit(new ClassPathResource("mnist_test.txt").getFile()));
        DataSetIterator iterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSizePerWorker, labelIndex, numClasses);
        List<DataSet> testDataList = new ArrayList<>();
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);

        ComputationGraph dis = new ComputationGraph(new NeuralNetConfiguration.Builder()
                                                                              .seed(666)
                                                                              .activation(Activation.RELU)
                                                                              .weightInit(WeightInit.XAVIER)
                                                                              .l2(learning_rate * 0.005) // regularize learning model
                                                                              .graphBuilder()
                                                                              .addInputs("dis_input_layer_0")
                                                                              .addLayer("dis_dense_layer_1", new DenseLayer.Builder().updater(new Sgd(frozen_learning_rate)).nIn(28 * 28).nOut(2000).build(), "dis_input_layer_0")
                                                                              .addLayer("dis_dense_layer_2", new DenseLayer.Builder().updater(new Sgd(frozen_learning_rate)).nIn(2000).nOut(2000).build(), "dis_dense_layer_1")
                                                                              .addLayer("dis_output_layer_3", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).updater(new Sgd(learning_rate)).nIn(2000).nOut(numClasses).activation(Activation.SOFTMAX).build(), "dis_dense_layer_2")
                                                                              .setOutputs("dis_output_layer_3")
                                                                              .build());
        dis.init();

        ComputationGraph gen = new ComputationGraph(new NeuralNetConfiguration.Builder()
                                                                              .seed(666)
                                                                              .activation(Activation.RELU)
                                                                              .weightInit(WeightInit.XAVIER)
                                                                              .l2(learning_rate * 0.005) // regularize learning model
                                                                              .graphBuilder()
                                                                              .addInputs("gen_input_layer_0")
                                                                              .addLayer("gen_dense_layer_1", new DenseLayer.Builder().updater(new Sgd(learning_rate)).nIn(28 * 28).nOut(2000).build(), "gen_input_layer_0")
                                                                              .addLayer("gen_dense_layer_2", new DenseLayer.Builder().updater(new Sgd(learning_rate)).nIn(2000).nOut(2000).build(), "gen_dense_layer_1")
                                                                              .addLayer("gen_output_layer_3", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).updater(new Sgd(learning_rate)).nIn(2000).nOut(numClasses).activation(Activation.SOFTMAX).build(), "gen_dense_layer_2")
                                                                              .setOutputs("gen_output_layer_3")
                                                                              .build());
        gen.init();

        System.out.println(dis.summary());
        System.out.println(gen.summary());

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
                                                                .averagingFrequency(5)
                                                                .workerPrefetchNumBatches(2)
                                                                .batchSizePerWorker(batchSizePerWorker)
                                                                .build();

        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, dis, tm);

        for (int i = 0; i < numEpochs; i++) {
            sparkNet.fit(trainData);
            log.info("Completed Epoch: {}.", i);
        }

        Evaluation evaluation = sparkNet.doEvaluation(testData, batchSizePerWorker, new Evaluation(numClasses))[0];
        log.info(evaluation.stats());

        gen.getLayer("gen_dense_layer_1").setParam("b", dis.getLayer("dis_dense_layer_1").getParam("b"));
        gen.getLayer("gen_dense_layer_1").setParam("W", dis.getLayer("dis_dense_layer_1").getParam("W"));
        gen.getLayer("gen_dense_layer_2").setParam("b", dis.getLayer("dis_dense_layer_2").getParam("b"));
        gen.getLayer("gen_dense_layer_2").setParam("W", dis.getLayer("dis_dense_layer_2").getParam("W"));
        gen.getLayer("gen_output_layer_3").setParam("b", dis.getLayer("dis_output_layer_3").getParam("b"));
        gen.getLayer("gen_output_layer_3").setParam("W", dis.getLayer("dis_output_layer_3").getParam("W"));

        SparkComputationGraph sparkNetGen = new SparkComputationGraph(sc, gen, tm);

        for (int i = 0; i < numEpochs; i++) {
            sparkNetGen.fit(trainData);
            log.info("Completed Epoch: {}.", i);
        }

        Evaluation evaluation_gen = sparkNetGen.doEvaluation(testData, batchSizePerWorker, new Evaluation(numClasses))[0];
        log.info(evaluation_gen.stats());

        tm.deleteTempFiles(sc);
    }
}
