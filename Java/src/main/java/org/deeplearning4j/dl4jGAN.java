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
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.*;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;

import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
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

    private static final int batchSizePerWorker = 10000;
    private static final int batchSizePred = 1;
    private static final int numEpochs = 1;
    private static final int numLinesToSkip = 0;
    private static final int labelIndex = 784;
    private static final int numClasses = 2;
    private static final int zSize = 2;

    private static final double learning_rate = 0.0015;
    private static final double frozen_learning_rate = 0.0;

    private static final String delimiter = ",";

    private static final boolean useGpu = false;

    public static void main(String[] args) throws Exception {
        new dl4jGAN().GAN(args);
    }

    private void GAN(String[] args) throws Exception {
        if (useGpu) {
            Nd4j.setDataType(DataBuffer.Type.FLOAT);

            CudaEnvironment.getInstance().getConfiguration()
                    .allowMultiGPU(true)
                    .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
                    .allowCrossDeviceAccess(true)
                    .setVerbose(true);
        }

        Nd4j.getRandom().setSeed(666);

        for (int i = 0; i < args.length; i++) {
            System.out.println(args[i]);
        }

        // Unfrozen discriminator.
        ComputationGraph dis = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(learning_rate * 0.005)
                .graphBuilder()
                .addInputs("dis_input_layer_0")
                .setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                .addLayer("dis_batch_layer_1", new BatchNormalization.Builder()
                        .updater(new Sgd(learning_rate))
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

        // Frozen generator.
        ComputationGraph gen = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(learning_rate * 0.005)
                .graphBuilder()
                .addInputs("gen_input_layer_0")
                .setInputTypes(InputType.feedForward(zSize))
                .addLayer("gen_batch_1", new BatchNormalization.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .build(),"gen_input_layer_0")
                .addLayer("gen_dense_layer_2", new DenseLayer.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .nOut(1024)
                        .build(),"gen_batch_1")
                .addLayer("gen_dense_layer_3", new DenseLayer.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .nOut(7 * 7 * 128)
                        .build(),"gen_dense_layer_2")
                .addLayer("gen_batch_4", new BatchNormalization.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .build(),"gen_dense_layer_3")
                .inputPreProcessor("gen_deconv2d_5", new FeedForwardToCnnPreProcessor(7, 7, 128))
                .addLayer("gen_deconv2d_5", new Deconvolution2D.Builder(6, 6)
                        .stride(2, 2)
                        .updater(new Sgd(frozen_learning_rate))
                        .nIn(128)
                        .nOut(64)
                        .build(),"gen_batch_4")
                .addLayer("gen_conv2d_6", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new Sgd(frozen_learning_rate))
                        .nIn(64)
                        .nOut(64)
                        .build(),"gen_deconv2d_5")
                .addLayer("gen_deconv2d_7", new Deconvolution2D.Builder(6, 6)
                        .stride(2, 2)
                        .updater(new Sgd(frozen_learning_rate))
                        .nIn(64)
                        .nOut(64)
                        .build(),"gen_conv2d_6")
                .addLayer("gen_conv2d_8", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new Sgd(frozen_learning_rate))
                        .nIn(64)
                        .nOut(1)
                        .build(),"gen_deconv2d_7")
                .addLayer("gen_output_layer_9", new DenseLayer.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .activation(Activation.SIGMOID)
                        .nOut(28 * 28)
                        .build(),"gen_conv2d_8")
                .setOutputs("gen_output_layer_9")
                .build());
        gen.init();
        System.out.println(gen.summary());

        // GAN with unfrozen generator and frozen discriminator.
        ComputationGraph gan = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(learning_rate * 0.005)
                .graphBuilder()
                .addInputs("gan_input_layer_0")
                .setInputTypes(InputType.feedForward(zSize))
                .addLayer("gan_batch_1", new BatchNormalization.Builder()
                        .updater(new Sgd(learning_rate))
                        .build(),"gan_input_layer_0")
                .addLayer("gan_dense_layer_2", new DenseLayer.Builder()
                        .updater(new Sgd(learning_rate))
                        .nOut(1024)
                        .build(),"gan_batch_1")
                .addLayer("gan_dense_layer_3", new DenseLayer.Builder()
                        .updater(new Sgd(learning_rate))
                        .nOut(7 * 7 * 128)
                        .build(),"gan_dense_layer_2")
                .addLayer("gan_batch_4", new BatchNormalization.Builder()
                        .updater(new Sgd(learning_rate))
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
                .addLayer("gan_dense_layer_9", new DenseLayer.Builder()
                        .updater(new Sgd(learning_rate))
                        .activation(Activation.SIGMOID)
                        .nOut(28 * 28)
                        .build(),"gan_conv2d_8")

                .addLayer("gan_dis_batch_layer_10", new BatchNormalization.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .build(),"gan_dense_layer_9")
                .inputPreProcessor("gan_dis_conv2d_layer_11", new FeedForwardToCnnPreProcessor(28, 28, 1))
                .addLayer("gan_dis_conv2d_layer_11", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new Sgd(frozen_learning_rate))
                        .nIn(1)
                        .nOut(64)
                        .build(),"gan_dis_batch_layer_10")
                .addLayer("gan_dis_maxpool_layer_12", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(),"gan_dis_conv2d_layer_11")
                .addLayer("gan_dis_conv2d_layer_13", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new Sgd(frozen_learning_rate))
                        .nIn(64)
                        .nOut(128)
                        .build(),"gan_dis_maxpool_layer_12")
                .addLayer("gan_dis_maxpool_layer_14", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(),"gan_dis_conv2d_layer_13")
                .addLayer("gan_dis_dense_layer_15", new DenseLayer.Builder()
                        .updater(new Sgd(frozen_learning_rate))
                        .nOut(1024)
                        .build(),"gan_dis_maxpool_layer_14")
                .addLayer("gan_dis_output_layer_16", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .updater(new Sgd(frozen_learning_rate))
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build(),"gan_dis_dense_layer_15")
                .setOutputs("gan_dis_output_layer_16")
                .build());
        gan.init();
        System.out.println(gan.summary());

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("DL4J Apache Spark: Generative Adversarial Network!");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
                .averagingFrequency(1000)
                .workerPrefetchNumBatches(0)
                .batchSizePerWorker(batchSizePerWorker)
                .build();

        SparkComputationGraph sparkDis = new SparkComputationGraph(sc, dis, tm);
        SparkComputationGraph sparkGan = new SparkComputationGraph(sc, gan, tm);

        RecordReader recordReaderTrain = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTrain.initialize(new FileSplit(new ClassPathResource("mnist_train.csv").getFile()));

        DataSetIterator iterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSizePerWorker, labelIndex, numClasses);

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            int batch_counter = 0;
            while (iterTrain.hasNext()) {
                List<DataSet> trainDataList = new ArrayList<>();
                trainDataList.add(iterTrain.next());
                for (int i = 0; i < batchSizePerWorker; i++) {
                    // [Fake, Real].
                    trainDataList.add(new DataSet(gen.output(Nd4j.randn(1, zSize))[0], Nd4j.hstack(Nd4j.ones(1, 1), Nd4j.zeros(1, 1))));
                }
                JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
                log.info("Training discriminator!");
                sparkDis.fit(trainData);

                // Update frozen discriminator with discriminator.
                gan.getLayer("gan_dis_batch_layer_10").setParam("gamma", dis.getLayer("dis_batch_layer_1").getParam("gamma"));
                gan.getLayer("gan_dis_batch_layer_10").setParam("beta", dis.getLayer("dis_batch_layer_1").getParam("beta"));
                gan.getLayer("gan_dis_batch_layer_10").setParam("mean", dis.getLayer("dis_batch_layer_1").getParam("mean"));
                gan.getLayer("gan_dis_batch_layer_10").setParam("var", dis.getLayer("dis_batch_layer_1").getParam("var"));

                gan.getLayer("gan_dis_conv2d_layer_11").setParam("W", dis.getLayer("dis_conv2d_layer_2").getParam("W"));
                gan.getLayer("gan_dis_conv2d_layer_11").setParam("b", dis.getLayer("dis_conv2d_layer_2").getParam("b"));

                gan.getLayer("gan_dis_conv2d_layer_13").setParam("W", dis.getLayer("dis_conv2d_layer_4").getParam("W"));
                gan.getLayer("gan_dis_conv2d_layer_13").setParam("b", dis.getLayer("dis_conv2d_layer_4").getParam("b"));

                gan.getLayer("gan_dis_dense_layer_15").setParam("W", dis.getLayer("dis_dense_layer_6").getParam("W"));
                gan.getLayer("gan_dis_dense_layer_15").setParam("b", dis.getLayer("dis_dense_layer_6").getParam("b"));

                gan.getLayer("gan_dis_output_layer_16").setParam("W", dis.getLayer("dis_output_layer_7").getParam("W"));
                gan.getLayer("gan_dis_output_layer_16").setParam("b", dis.getLayer("dis_output_layer_7").getParam("b"));

                trainDataList = new ArrayList<>();
                for (int i = 0; i < batchSizePerWorker; i++) {
                    // Tell the frozen discriminator that all the fake examples are real examples.
                    // [Fake, Real].
                    trainDataList.add(new DataSet(Nd4j.randn(1, zSize), Nd4j.hstack(Nd4j.zeros(1, 1), Nd4j.ones(1, 1))));
                }
                trainData = sc.parallelize(trainDataList);
                log.info("Training generator!");
                sparkGan.fit(trainData);

                // Update generator with GAN's generator.
                gen.getLayer("gen_batch_1").setParam("gamma", gan.getLayer("gan_batch_1").getParam("gamma"));
                gen.getLayer("gen_batch_1").setParam("beta", gan.getLayer("gan_batch_1").getParam("beta"));
                gen.getLayer("gen_batch_1").setParam("mean", gan.getLayer("gan_batch_1").getParam("mean"));
                gen.getLayer("gen_batch_1").setParam("var", gan.getLayer("gan_batch_1").getParam("var"));

                gen.getLayer("gen_dense_layer_2").setParam("W", gan.getLayer("gan_dense_layer_2").getParam("W"));
                gen.getLayer("gen_dense_layer_2").setParam("b", gan.getLayer("gan_dense_layer_2").getParam("b"));

                gen.getLayer("gen_dense_layer_3").setParam("W", gan.getLayer("gan_dense_layer_3").getParam("W"));
                gen.getLayer("gen_dense_layer_3").setParam("b", gan.getLayer("gan_dense_layer_3").getParam("b"));

                gen.getLayer("gen_batch_4").setParam("gamma", gan.getLayer("gan_batch_4").getParam("gamma"));
                gen.getLayer("gen_batch_4").setParam("beta", gan.getLayer("gan_batch_4").getParam("beta"));
                gen.getLayer("gen_batch_4").setParam("mean", gan.getLayer("gan_batch_4").getParam("mean"));
                gen.getLayer("gen_batch_4").setParam("var", gan.getLayer("gan_batch_4").getParam("var"));

                gen.getLayer("gen_deconv2d_5").setParam("W", gan.getLayer("gan_deconv2d_5").getParam("W"));
                gen.getLayer("gen_deconv2d_5").setParam("b", gan.getLayer("gan_deconv2d_5").getParam("b"));

                gen.getLayer("gen_conv2d_6").setParam("W", gan.getLayer("gan_conv2d_6").getParam("W"));
                gen.getLayer("gen_conv2d_6").setParam("b", gan.getLayer("gan_conv2d_6").getParam("b"));

                gen.getLayer("gen_deconv2d_7").setParam("W", gan.getLayer("gan_deconv2d_7").getParam("W"));
                gen.getLayer("gen_deconv2d_7").setParam("b", gan.getLayer("gan_deconv2d_7").getParam("b"));

                gen.getLayer("gen_conv2d_8").setParam("W", gan.getLayer("gan_conv2d_8").getParam("W"));
                gen.getLayer("gen_conv2d_8").setParam("b", gan.getLayer("gan_conv2d_8").getParam("b"));

                gen.getLayer("gen_output_layer_9").setParam("W", gan.getLayer("gan_dense_layer_9").getParam("W"));
                gen.getLayer("gen_output_layer_9").setParam("b", gan.getLayer("gan_dense_layer_9").getParam("b"));

                log.info("Completed Batch {}!", batch_counter + 1);
                batch_counter++;
            }
            log.info("Completed Epoch {}!", epoch + 1);
        }

        RecordReader recordReaderTest = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTest.initialize(new FileSplit(new ClassPathResource("mnist_test.csv").getFile()));

        DataSetIterator iterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSizePred, labelIndex, numClasses);

        List<DataSet> testDataList = new ArrayList<>();
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        JavaRDD<DataSet> testData = sc.parallelize(testDataList);

        log.info("Number of testing examples: {}!", testDataList.size());

        Evaluation evaluation = sparkDis.doEvaluation(testData, batchSizePred, new Evaluation(numClasses))[0];
        log.info(evaluation.stats());

        int counter = 0;
        INDArray testDataPred = Nd4j.zeros(testDataList.size(), numClasses);
        Iterator<DataSet> testDataListIter = testDataList.iterator();
        while (testDataListIter.hasNext()) {
            testDataPred.putRow(counter, sparkDis.getNetwork().output(testDataListIter.next().getFeatureMatrix())[0]);
            counter++;
        }
        Nd4j.writeNumpy(testDataPred, "/Users/samson/Projects/gan_deeplearning4j/Java/src/main/resources/testDataPredMnist.csv", ",");

        tm.deleteTempFiles(sc);
    }
}
