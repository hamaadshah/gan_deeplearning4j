package org.deeplearning4j;

import java.util.*;
import java.util.concurrent.TimeUnit;

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
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
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
    private static final int numEpochs = 10;
    private static final int numLinesToSkip = 0;
    private static final int labelIndex = 784;
    private static final int numClasses = 2;
    private static final int zSize = 2;

    private static final double learning_rate = 0.0015;
    private static final double frozen_learning_rate = 0.0;

    private static final String delimiter = ",";

    public static void main(String[] args) throws Exception {
        new dl4jGAN().GAN(args);
    }

    private void GAN(String[] args) throws Exception {
        Nd4j.getRandom().setSeed(666);

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
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(learning_rate * 0.005)
                .graphBuilder()
                .addInputs("gen_input_layer_0")
                .setInputTypes(InputType.feedForward(zSize))
                .addLayer("gen_batch_1", new BatchNormalization.Builder()
                        .build(),"gen_input_layer_0")
                .addLayer("gen_dense_layer_2", new DenseLayer.Builder()
                        .updater(new Sgd(learning_rate))
                        .nOut(1024)
                        .build(),"gen_batch_1")
                .addLayer("gen_dense_layer_3", new DenseLayer.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .nOut(7 * 7 * 128)
                        .build(),"gen_dense_layer_2")
                .addLayer("gen_batch_4", new BatchNormalization.Builder()
                        .build(),"gen_dense_layer_3")
                .inputPreProcessor("gen_deconv2d_5", new FeedForwardToCnnPreProcessor(7, 7, 128))
                .addLayer("gen_deconv2d_5", new Deconvolution2D.Builder(6, 6)
                        .stride(2, 2)
                        .updater(new Sgd(learning_rate))
                        .nIn(128)
                        .nOut(64)
                        .build(),"gen_batch_4")
                .addLayer("gen_conv2d_6", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new Sgd(learning_rate))
                        .nIn(64)
                        .nOut(64)
                        .build(),"gen_deconv2d_5")
                .addLayer("gen_deconv2d_7", new Deconvolution2D.Builder(6, 6)
                        .stride(2, 2)
                        .updater(new Sgd(learning_rate))
                        .nIn(64)
                        .nOut(64)
                        .build(),"gen_conv2d_6")
                .addLayer("gen_conv2d_8", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new Sgd(learning_rate))
                        .nIn(64)
                        .nOut(1)
                        .build(),"gen_deconv2d_7")
                .addLayer("gen_output_layer_9", new DenseLayer.Builder()
                        .updater(new Sgd(learning_rate))
                        .activation(Activation.SIGMOID)
                        .nOut(28 * 28)
                        .build(),"gen_conv2d_8")
                .setOutputs("gen_output_layer_9")
                .build());
        gen.init();
        System.out.println(gen.summary());

        ComputationGraph gan = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(learning_rate * 0.005)
                .graphBuilder()
                .addInputs("gan_input_layer_0")
                .setInputTypes(InputType.feedForward(zSize))
                .addLayer("gan_batch_1", new BatchNormalization.Builder()
                        .build(),"gan_input_layer_0")
                .addLayer("gan_dense_layer_2", new DenseLayer.Builder()
                        .updater(new Sgd(learning_rate))
                        .nOut(1024)
                        .build(),"gan_batch_1")
                .addLayer("gan_dense_layer_3", new DenseLayer.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .nOut(7 * 7 * 128)
                        .build(),"gan_dense_layer_2")
                .addLayer("gan_batch_4", new BatchNormalization.Builder()
                        .build(),"gan_dense_layer_3")
                .inputPreProcessor("gan_deconv2d_5", new FeedForwardToCnnPreProcessor(7, 7, 128))
                .addLayer("gan_deconv2d_5", new Deconvolution2D.Builder(6, 6)
                        .stride(2, 2)
                        .updater(new Sgd(learning_rate))
                        .nIn(128)
                        .nOut(64)
                        .build(),"gan_batch_4")
                .addLayer("gan_conv2d_6", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new Sgd(learning_rate))
                        .nIn(64)
                        .nOut(64)
                        .build(),"gan_deconv2d_5")
                .addLayer("gan_deconv2d_7", new Deconvolution2D.Builder(6, 6)
                        .stride(2, 2)
                        .updater(new Sgd(learning_rate))
                        .nIn(64)
                        .nOut(64)
                        .build(),"gan_conv2d_6")
                .addLayer("gan_conv2d_8", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new Sgd(learning_rate))
                        .nIn(64)
                        .nOut(1)
                        .build(),"gan_deconv2d_7")

                .addLayer("gan_dis_batch_layer_9", new BatchNormalization.Builder()
                        .build(),"gan_conv2d_8")
                .addLayer("gan_dis_conv2d_layer_10", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new Sgd(learning_rate))
                        .nIn(1)
                        .nOut(64)
                        .build(),"gan_dis_batch_layer_9")
                .addLayer("gan_dis_maxpool_layer_11", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(),"gan_dis_conv2d_layer_10")
                .addLayer("gan_dis_conv2d_layer_12", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new Sgd(learning_rate))
                        .nIn(64)
                        .nOut(128)
                        .build(),"gan_dis_maxpool_layer_11")
                .addLayer("gan_dis_maxpool_layer_13", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(),"gan_dis_conv2d_layer_12")
                .addLayer("gan_dis_dense_layer_14", new DenseLayer.Builder()
                        .updater(new Sgd(learning_rate))
                        .nOut(1024)
                        .build(),"gan_dis_maxpool_layer_13")
                .addLayer("gan_dis_output_layer_15", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .updater(new Sgd(learning_rate))
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build(),"gan_dis_dense_layer_14")
                .setOutputs("gan_dis_output_layer_15")
                .build());
        gan.init();
        System.out.println(gan.summary());

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("DL4J Apache Spark: Generative Adversarial Network");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        RecordReader recordReaderTrain = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTrain.initialize(new FileSplit(new ClassPathResource("mnist_train.csv").getFile()));

        DataSetIterator iterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSizePerWorker, labelIndex, numClasses);

        List<DataSet> trainDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }

        int numTrainPred = trainDataList.size() * batchSizePerWorker;
        log.info("Number of training examples: {}.", numTrainPred);

        for (int i = 0; i < batchSizePerWorker; i++) {
            trainDataList.add(new DataSet(gen.output(Nd4j.randn(1, zSize))[0], Nd4j.hstack(Nd4j.ones(1,1), Nd4j.zeros(1,1))));
        }

        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);

        RecordReader recordReaderTest = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTest.initialize(new FileSplit(new ClassPathResource("mnist_test.csv").getFile()));

        DataSetIterator iterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSizePred, labelIndex, numClasses);

        List<DataSet> testDataList = new ArrayList<>();
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        int numTestPred = testDataList.size() * batchSizePred;
        log.info("Number of testing examples: {}.", numTestPred);

        for (int i = 0; i < batchSizePerWorker; i++) {
            testDataList.add(new DataSet(gen.output(Nd4j.randn(1, zSize))[0], Nd4j.hstack(Nd4j.ones(1,1), Nd4j.zeros(1,1))));
        }

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

        int counter = 0;
        INDArray testDataPred = Nd4j.zeros(numTestPred + batchSizePerWorker, numClasses);
        Iterator<DataSet> testDataListIter = testDataList.iterator();
        while (testDataListIter.hasNext()) {
            testDataPred.putRow(counter, sparkNet.getNetwork().output(testDataListIter.next().getFeatureMatrix())[0]);
            counter++;
        }
        Nd4j.writeNumpy(testDataPred, "/Users/samson/Projects/gan_deeplearning4j/Java/src/main/resources/testDataPredMnist.csv", ",");

        tm.deleteTempFiles(sc);
    }
}
