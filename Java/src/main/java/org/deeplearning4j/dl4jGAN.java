package org.deeplearning4j;

import java.util.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.storage.StorageLevel;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.*;
import org.deeplearning4j.nn.weights.*;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;

import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
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
    private static final int batchSizePred = 50;
    private static final int numIterations = 10000;
    private static final int numLinesToSkip = 0;
    private static final int labelIndex = 784;
    private static final int numClasses = 10;
    private static final int zSize = 2;

    private static final double dis_learning_rate = 0.025;
    private static final double gen_learning_rate = 0.05;
    private static final double frozen_learning_rate = 0.0;

    private static final String delimiter = ",";
    private static final String resPath = "/Users/samson/Projects/gan_deeplearning4j/Java/src/main/resources/";

    private static final boolean useGpu = false;

    public static void main(String[] args) throws Exception {
        new dl4jGAN().GAN(args);
    }

    private void GAN(String[] args) throws Exception {
        for (int i = 0; i < args.length; i++) {
            System.out.println(args[i]);
        }

        if (useGpu) {
            Nd4j.setDataType(DataBuffer.Type.FLOAT);

            CudaEnvironment.getInstance().getConfiguration()
                    .allowMultiGPU(true)
                    .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
                    .allowCrossDeviceAccess(true)
                    .setVerbose(true);
        }

        Nd4j.getRandom().setSeed(666);
        Nd4j.getMemoryManager().togglePeriodicGc(true);
        System.out.println(Nd4j.getBackend());

        // Unfrozen discriminator.
        ComputationGraph dis = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(dis_learning_rate * 0.005)
                .graphBuilder()
                .addInputs("dis_input_layer_0")
                .setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                .addLayer("dis_batch_layer_1", new BatchNormalization.Builder()
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .build(), "dis_input_layer_0")
                .addLayer("dis_conv2d_layer_2", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nIn(1)
                        .nOut(64)
                        .build(), "dis_batch_layer_1")
                .addLayer("dis_maxpool_layer_3", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "dis_conv2d_layer_2")
                .addLayer("dis_conv2d_layer_4", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(128)
                        .build(), "dis_maxpool_layer_3")
                .addLayer("dis_maxpool_layer_5", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "dis_conv2d_layer_4")
                .addLayer("dis_dense_layer_6", new DenseLayer.Builder()
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "dis_maxpool_layer_5")
                .addLayer("dis_output_layer_7", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build(), "dis_dense_layer_6")
                .setOutputs("dis_output_layer_7")
                .build());
        dis.init();
        System.out.println(dis.summary());

        // Frozen generator.
        ComputationGraph gen = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(frozen_learning_rate * 0.005)
                .graphBuilder()
                .addInputs("gen_input_layer_0")
                .setInputTypes(InputType.feedForward(zSize))
                .addLayer("gen_batch_1", new BatchNormalization.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .build(), "gen_input_layer_0")
                .addLayer("gen_dense_layer_2", new DenseLayer.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "gen_batch_1")
                .addLayer("gen_dense_layer_3", new DenseLayer.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(7 * 7 * 128)
                        .build(), "gen_dense_layer_2")
                .addLayer("gen_batch_4", new BatchNormalization.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .build(), "gen_dense_layer_3")
                .inputPreProcessor("gen_deconv2d_5", new FeedForwardToCnnPreProcessor(7, 7, 128))
                .addLayer("gen_deconv2d_5", new Upsampling2D.Builder(2)
                        .build(), "gen_batch_4")
                .addLayer("gen_conv2d_6", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(128)
                        .nOut(64)
                        .build(), "gen_deconv2d_5")
                .addLayer("gen_deconv2d_7", new Upsampling2D.Builder(2)
                        .build(), "gen_conv2d_6")
                .addLayer("gen_conv2d_8", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(1)
                        .build(), "gen_deconv2d_7")
                .addLayer("gen_output_layer_9", new DenseLayer.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .activation(Activation.SIGMOID)
                        .nOut(28 * 28)
                        .build(), "gen_conv2d_8")
                .setOutputs("gen_output_layer_9")
                .build());
        gen.init();
        System.out.println(gen.summary());

        // GAN with unfrozen generator and frozen discriminator.
        ComputationGraph gan = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(gen_learning_rate * 0.005)
                .graphBuilder()
                .addInputs("gan_input_layer_0")
                .setInputTypes(InputType.feedForward(zSize))
                .addLayer("gan_batch_1", new BatchNormalization.Builder()
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .build(), "gan_input_layer_0")
                .addLayer("gan_dense_layer_2", new DenseLayer.Builder()
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "gan_batch_1")
                .addLayer("gan_dense_layer_3", new DenseLayer.Builder()
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .nOut(7 * 7 * 128)
                        .build(), "gan_dense_layer_2")
                .addLayer("gan_batch_4", new BatchNormalization.Builder()
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .build(), "gan_dense_layer_3")
                .inputPreProcessor("gan_deconv2d_5", new FeedForwardToCnnPreProcessor(7, 7, 128))
                .addLayer("gan_deconv2d_5", new Upsampling2D.Builder(2)
                        .build(), "gan_batch_4")
                .addLayer("gan_conv2d_6", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .nIn(128)
                        .nOut(64)
                        .build(), "gan_deconv2d_5")
                .addLayer("gan_deconv2d_7", new Upsampling2D.Builder(2)
                        .build(), "gan_conv2d_6")
                .addLayer("gan_conv2d_8", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(1)
                        .build(), "gan_deconv2d_7")
                .addLayer("gan_dense_layer_9", new DenseLayer.Builder()
                        .updater(new RmsProp(gen_learning_rate, 1e-8, 1e-8))
                        .activation(Activation.SIGMOID)
                        .nOut(28 * 28)
                        .build(), "gan_conv2d_8")

                .addLayer("gan_dis_batch_layer_10", new BatchNormalization.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .build(), "gan_dense_layer_9")
                .inputPreProcessor("gan_dis_conv2d_layer_11", new FeedForwardToCnnPreProcessor(28, 28, 1))
                .addLayer("gan_dis_conv2d_layer_11", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(1)
                        .nOut(64)
                        .build(), "gan_dis_batch_layer_10")
                .addLayer("gan_dis_maxpool_layer_12", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "gan_dis_conv2d_layer_11")
                .addLayer("gan_dis_conv2d_layer_13", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(128)
                        .build(), "gan_dis_maxpool_layer_12")
                .addLayer("gan_dis_maxpool_layer_14", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "gan_dis_conv2d_layer_13")
                .addLayer("gan_dis_dense_layer_15", new DenseLayer.Builder()
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "gan_dis_maxpool_layer_14")
                .addLayer("gan_dis_output_layer_16", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .updater(new RmsProp(frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build(), "gan_dis_dense_layer_15")
                .setOutputs("gan_dis_output_layer_16")
                .build());
        gan.init();
        System.out.println(gan.summary());

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("DL4J Apache Spark: Generative Adversarial Network!");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
                .averagingFrequency(10)
                .rngSeed(666)
                .storageLevel(StorageLevel.DISK_ONLY())
                .workerPrefetchNumBatches(0)
                .batchSizePerWorker(batchSizePerWorker)
                .build();

        SparkComputationGraph sparkDis = new SparkComputationGraph(sc, dis, tm);
        SparkComputationGraph sparkGan = new SparkComputationGraph(sc, gan, tm);

        RecordReader recordReaderTrain = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTrain.initialize(new FileSplit(new ClassPathResource("mnist_train.csv").getFile()));

        DataSetIterator iterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSizePerWorker, labelIndex, numClasses);
        List<DataSet> trainDataList = new ArrayList<>();

        JavaRDD<DataSet> trainDataDis, trainDataGen;
        INDArray soften_labels_fake, soften_labels_real;
        INDArray out;

        int batch_counter = 0;
        while (iterTrain.hasNext() && batch_counter < numIterations) {
            trainDataList.clear();
            // This is real data...
            // [Fake, Real].
            soften_labels_fake = Nd4j.randn(batchSizePerWorker, 1).muli(0.05);
            soften_labels_real = Nd4j.randn(batchSizePerWorker, 1).muli(0.05);
            trainDataList.add(new DataSet(iterTrain.next().getFeatureMatrix(), Nd4j.hstack(Nd4j.zeros(batchSizePerWorker, 1).addi(soften_labels_fake), Nd4j.ones(batchSizePerWorker, 1).addi(soften_labels_real))));

            // ...and this is fake data.
            // [Fake, Real].
            soften_labels_fake = Nd4j.randn(batchSizePerWorker, 1).muli(0.05);
            soften_labels_real = Nd4j.randn(batchSizePerWorker, 1).muli(0.05);
            trainDataList.add(new DataSet(gen.output(Nd4j.rand(batchSizePerWorker, zSize).muli(2.0).subi(1.0))[0], Nd4j.hstack(Nd4j.ones(batchSizePerWorker, 1).addi(soften_labels_fake), Nd4j.zeros(batchSizePerWorker, 1).addi(soften_labels_real))));

            log.info("Training discriminator!");
            trainDataDis = sc.parallelize(trainDataList).persist(StorageLevel.DISK_ONLY());
            sparkDis.fit(trainDataDis);

            // Update GAN's frozen discriminator with unfrozen discriminator.
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

            trainDataList.clear();
            // Tell the frozen discriminator that all the fake examples are real examples.
            // [Fake, Real].
            trainDataList.add(new DataSet(Nd4j.rand(batchSizePerWorker, zSize).muli(2.0).subi(1.0), Nd4j.hstack(Nd4j.zeros(batchSizePerWorker, 1), Nd4j.ones(batchSizePerWorker, 1))));

            log.info("Training generator!");
            trainDataGen = sc.parallelize(trainDataList).persist(StorageLevel.DISK_ONLY());
            sparkGan.fit(trainDataGen);

            // Update frozen generator with GAN's unfrozen generator.
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

            gen.getLayer("gen_conv2d_6").setParam("W", gan.getLayer("gan_conv2d_6").getParam("W"));
            gen.getLayer("gen_conv2d_6").setParam("b", gan.getLayer("gan_conv2d_6").getParam("b"));

            gen.getLayer("gen_conv2d_8").setParam("W", gan.getLayer("gan_conv2d_8").getParam("W"));
            gen.getLayer("gen_conv2d_8").setParam("b", gan.getLayer("gan_conv2d_8").getParam("b"));

            gen.getLayer("gen_output_layer_9").setParam("W", gan.getLayer("gan_dense_layer_9").getParam("W"));
            gen.getLayer("gen_output_layer_9").setParam("b", gan.getLayer("gan_dense_layer_9").getParam("b"));

            log.info("Completed Batch {}!", batch_counter + 1);
            batch_counter += batchSizePerWorker;

            out = gen.output(Nd4j.rand(16, 2).muli(2.0).subi(1.0))[0];
            Nd4j.writeNumpy(out, String.format("%sout_%d.csv", resPath, batch_counter), delimiter);

            if (!iterTrain.hasNext() && batch_counter < numIterations) {
                iterTrain.reset();
            }
        }

        RecordReader recordReaderTest = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReaderTest.initialize(new FileSplit(new ClassPathResource("mnist_test.csv").getFile()));

        DataSetIterator iterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSizePred, labelIndex, numClasses);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(dis_learning_rate, 1e-8, 1e-8))
                .seed(666)
                .build();

        ComputationGraph featureExtractor = new TransferLearning.GraphBuilder(dis)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("dis_dense_layer_6")
                .removeVertexKeepConnections("dis_output_layer_7")
                .addLayer("dis_output_layer_7", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(1024)
                        .nOut(numClasses)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build(), "dis_dense_layer_6")
                .build();

        Collection<INDArray> outFeat = new ArrayList<>();
        while (iterTest.hasNext()) {
            outFeat.add(featureExtractor.output(iterTest.next().getFeatureMatrix())[0]);
        }
        Nd4j.writeNumpy(Nd4j.vstack(outFeat), resPath + "out_feat.csv", delimiter);

        tm.deleteTempFiles(sc);
    }
}
