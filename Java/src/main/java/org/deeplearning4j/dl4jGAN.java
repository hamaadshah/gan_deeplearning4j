package org.deeplearning4j;

import java.util.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;


import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.*;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;

import org.nd4j.linalg.activations.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class dl4jGAN {
    private static final Logger log = LoggerFactory.getLogger(dl4jGAN.class);

    private static final int batchSizePerWorker = 100;
    private static final int batchSizePred = 1;
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

        ComputationGraph dis = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(learning_rate * 0.005)
                .graphBuilder()
                .addInputs("dis_input_layer_0")
                .setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                .addLayer("dis_batch_layer_1", new BatchNormalization.Builder()
                        .build(),"dis_input_layer_0")
                .addLayer("dis_conv2d_layer_2", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new Sgd(learning_rate))
                        .nIn(1)
                        .nOut(64)
                        .build(),"dis_batch_layer_1")
                .addLayer("dis_maxpool_layer_3", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(),"dis_conv2d_layer_2")
                .addLayer("dis_conv2d_layer_4", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new Sgd(learning_rate))
                        .nIn(64)
                        .nOut(128)
                        .build(),"dis_maxpool_layer_3")
                .addLayer("dis_maxpool_layer_5", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(),"dis_conv2d_layer_4")
                .addLayer("dis_dense_layer_6", new DenseLayer.Builder()
                        .updater(new Sgd(learning_rate))
                        .nOut(1024)
                        .build(),"dis_maxpool_layer_5")
                .addLayer("dis_output_layer_7", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .updater(new Sgd(learning_rate))
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build(),"dis_dense_layer_6")
                .setOutputs("dis_output_layer_7")
                .build());
        dis.init();
        System.out.println(dis.summary());

        ComputationGraph gen = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .activation(Activation.ELU)
                .weightInit(WeightInit.XAVIER)
                .l2(learning_rate * 0.005)
                .graphBuilder()
                .addInputs("gen_input_layer_0")
                .setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                .addLayer("gen_batch_1", new BatchNormalization.Builder()
                        .build(),"gen_input_layer_0")
                .addLayer("gen_conv2d_layer", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new Sgd(learning_rate))
                        .nIn(1)
                        .nOut(1)
                        .build(),"gen_batch_1")
                .addLayer("gen_maxpool_layer", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(),"gen_conv2d_layer")
                .addLayer("gen_dense_layer_1", new DenseLayer.Builder()
                        .updater(new Sgd(learning_rate))
                        .nOut(2000)
                        .build(),"gen_maxpool_layer")
                .addLayer("gen_dense_layer_2", new DenseLayer.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .nOut(2000)
                        .build(),"gen_dense_layer_1")
                .addLayer("gen_output_layer_3", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .updater(new Sgd(learning_rate))
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build(),"gen_dense_layer_2")
                .setOutputs("gen_output_layer_3")
                .build());
        gen.init();
        System.out.println(gen.summary());

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("DL4J Apache Spark: Generative Adversarial Network");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        RecordReader recordReaderTrain = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTrain.initialize(new FileSplit(new ClassPathResource("mnist_train.txt").getFile()));
        DataSetIterator iterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSizePerWorker, labelIndex, numClasses);
        List<DataSet> trainDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }
        int numTrainPred = trainDataList.size() * batchSizePerWorker;
        System.out.println(numTrainPred);
        iterTrain.reset();
        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);

        RecordReader recordReaderTest = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTest.initialize(new FileSplit(new ClassPathResource("mnist_test.txt").getFile()));
        DataSetIterator iterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSizePred, labelIndex, numClasses);
        List<DataSet> testDataList = new ArrayList<>();
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }
        int numTestPred = testDataList.size() * batchSizePred;
        System.out.println(numTestPred);
        iterTest.reset();
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);

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
/*
        gen.getLayer("gen_batch_1").setParam("gamma", dis.getLayer("dis_batch_1").getParam("gamma"));
        gen.getLayer("gen_batch_1").setParam("beta", dis.getLayer("dis_batch_1").getParam("beta"));
        gen.getLayer("gen_batch_1").setParam("mean", dis.getLayer("dis_batch_1").getParam("mean"));
        gen.getLayer("gen_batch_1").setParam("var", dis.getLayer("dis_batch_1").getParam("var"));
        gen.getLayer("gen_conv2d_layer").setParam("b", dis.getLayer("dis_conv2d_layer").getParam("b"));
        gen.getLayer("gen_conv2d_layer").setParam("W", dis.getLayer("dis_conv2d_layer").getParam("W"));
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
*/

        int counter = 0;
        INDArray testDataPred = Nd4j.zeros(numTestPred, numClasses);
        while (iterTest.hasNext()) {
            testDataPred.putRow(counter, sparkNet.getNetwork().output(iterTest.next().getFeatureMatrix())[0]);
            counter++;
        }
        Nd4j.writeNumpy(testDataPred, "testDataPredMnist.csv", ",");

        tm.deleteTempFiles(sc);
    }
}
