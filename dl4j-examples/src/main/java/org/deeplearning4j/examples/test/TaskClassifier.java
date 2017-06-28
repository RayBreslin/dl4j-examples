package org.deeplearning4j.examples.test;

/*
 * ************************ Java-SDK *******************************
 */
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/*
 * ************************* Misc **********************************
 */
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/*
 * *************************************************************************************
 * *************************************************************************************
 */
public class TaskClassifier {
    /*
     * *****************************************************************
     * *********************** VARIABLES *******************************
     * *****************************************************************
     */
    /******************************************
     * FINAL VARIABLES
     *****************************************/
    // directory for the neural-network-model
    private static final String NN_MODEL_DIRECTORY = "dl4j-examples/src/main/resources/";
    // file for the neural-network-model
    private static final String NN_MODEL_FILE = "trained_nn_model.zip";

    private static final boolean MODEL_LOADING_ENABLED = false;
    private static final boolean MODEL_SAVING_ENABLED = true;

    // ui-configuration
    private static final boolean UI_ENABLED = false;

    // number of total data-sets in csv-file
    private static final int SAMPLE_NUMBER = 75000;
    // index (column) for output-labels in csv-file
    private static final int LABEL_INDEX = 48;
    // number of classes to classify (f.ex. number of activities)
    private static final int CLASS_NUMBER = 5;

    // input-layer
    private static final int LAYER_0_NEURONS = 48;
    // hidden-layer
    private static final int LAYER_1_NEURONS = 48;
    // output-layer
    private static final int LAYER_2_NEURONS = 5;

    private static final int MAX_ITERATIONS = 1;
    private static final double LEARNING_RATE = 0.001;

    private static final int MAX_EPOCHS = 5;

    // CSV-data: 6
    private static final long SEED_NUMBER = 6;

    private static final boolean TEST_DATA_LABELLED = true;

    /******************************************
     * STATIC VARIABLES
     *****************************************/
    // file that holds the neural-network-model
    protected static File sMLNetworkFile;

    private static MultiLayerNetwork sMLNetwork;

    private static MultiLayerConfiguration sMLNetworkConfiguration;

    private static DataSet sTrainingData;
    private static DataSet sTestData;

    private static DataSetIterator sTrainingDataIterator;
    private static DataSetIterator sTestDataIterator;

    private static DataNormalization sDataNormalizer;

    // re-training-permission for the neural-network-model after loading
    private static boolean sSaveUpdater = false;

    private static int sClassificationStatus;

    /******************************************
     * PRIVATE VARIABLES
     *****************************************/

	/*
	 * *****************************************************************
	 * ******************** MAIN - FUNCTIONS ***************************
	 * *****************************************************************
	 */
    private static void initializeNetwork() {
        Layer[] layers;
        int nParams = 0, totalNumParams = 0;
        long time1, time2;

        System.out.println("Initializing neural-network...");

        time1 = System.currentTimeMillis();

        sMLNetworkConfiguration = new NeuralNetConfiguration.Builder()
            // set how often should the training set be run (empirical value)
            // note: value > 1000 recommended
            .iterations(MAX_ITERATIONS)
            // set the learning rate (empirical value)
            // note: needs to be higher if the number of iterations is smaller
            .learningRate(LEARNING_RATE)
            // fixed the seed for the random generator (any run of this program brings the same esResults)
            // note: may not work if you do something like ds.shuffle()
            .seed(SEED_NUMBER)
            // put the layer output through an activation function to adjust the output
            // note: tanh function caps the output to (-1,1)
            .activation(Activation.RELU)
            // initialize the connection weights
            // note: xavier initializes weights with mean 0 and variance 2/(fanIn + fanOut)
            .weightInit(WeightInit.XAVIER)
            // initialize the connection bias (empirical value)
            //.biasInit(0)
            // not applicable, this network is to small - but for bigger networks it
            // can help that the network will not only recite the training data
            //.useDropConnect(false)
            // set the optimization-algorithm for moving on the error-plane
            // (usage of LINE_GRADIENT_DESCENT or CONJUGATE_GRADIENT pretty much equivalent)
            // note: configuration might need to get a project-specific adjustment
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            // disable mini-batch-processing
            // (usage not recommended if the data-set is smaller than the mini batch size)
            // note: networks can process the input more quickly and more accurately by ingesting
            // minibatches 5-10 elements at a time in parallel
            // (see http://deeplearning4j.org/architecture)
            //.miniBatch(false)
            .regularization(true).l2(1e-4)
            .list()
            // create a multilayer network with 3 layers
            // note: local layer-settings override global layer-settings
            // 0. input layer
            .layer(0, new DenseLayer.Builder()
                // define the number of input connections
                .nIn(LAYER_0_NEURONS)
                // define the number of outgoing connections
                .nOut(LAYER_1_NEURONS)
                .build())
            // 1. hidden layer
            .layer(1, new DenseLayer.Builder()
                .nIn(LAYER_1_NEURONS)
                .nOut(LAYER_1_NEURONS)
                .build())
            // 2. output layer
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(LAYER_1_NEURONS)
                .nOut(LAYER_2_NEURONS)
                .build())
            // skip pre-train phase for this network
            .pretrain(false)
            // use back-propagation
            // (typically used with pretrain(true) for finetuning without changing layer-weights)
            .backprop(true)
            .build();

        // init the network, will check if everything is configured correct
        sMLNetwork = new MultiLayerNetwork(sMLNetworkConfiguration);
        sMLNetwork.init();
        time2 = System.currentTimeMillis();

        if (UI_ENABLED == true) {
            initializeUI();
        } else {
            // add a listener which outputs the error every 100 parameter updates
            sMLNetwork.setListeners(new ScoreIterationListener(100));
        }// if (UI_ENABLED == true)

        // print the number of parameters in the network (and for each layer)
        layers = sMLNetwork.getLayers();
        for (int i = 0; i < layers.length; i++) {
            nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }// for (int i = 0; i < layers.length; i++)
        System.out.println("Total number of network parameters: " + totalNumParams);

        System.out.println("sMLNetwork.init(): " + (time2 - time1));

        System.out.println("Initialization completed!");
    }

	private static void initializeUI() {
        // initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        // configure the memory-storage for network information (gradients, score vs. time etc)
        // (alternative: new FileStatsStorage(File), for saving and loading later)
        StatsStorage statsStorage = new InMemoryStatsStorage();

        // attach the StatsStorage instance to the UI
        // (this allows the contents of the StatsStorage to be visualized)
        uiServer.attach(statsStorage);

        // add the StatsListener to collect this information from the network, as it trains
        sMLNetwork.setListeners(new StatsListener(statsStorage));
    }

    private static void trainNetwork() {
        long time1, time2;

        System.out.println("Training neural-network...");
        time1 = System.currentTimeMillis();

        if (MODEL_LOADING_ENABLED == true) {
            loadData("task_data_75000_training.csv", 0, SAMPLE_NUMBER, 512, LABEL_INDEX, CLASS_NUMBER);
            loadNetwork();
        } else {
            loadData("task_data_75000_training.csv", 0, SAMPLE_NUMBER, 512, LABEL_INDEX, CLASS_NUMBER);

            // train the network
            for (int i = 0; i < MAX_EPOCHS; i++) {
                sMLNetwork.fit(sTrainingDataIterator);
            }// for (int i = 0; i < MAX_EPOCHS; i++)
            time2 = System.currentTimeMillis();
            System.out.println("sMLNetwork.fit(): " + (time2 - time1));
        }// if (MODEL_LOADING_ENABLED == true)

        if (MODEL_SAVING_ENABLED == true) {
            saveNetwork();
        }// if (MODEL_SAVING_ENABLED == true)

        System.out.println("Training completed!");
    }

    private static void testNetwork() {
        INDArray prediction;
        INDArray mean;
        Evaluation eval;
        long time1, time2, time3;

        System.out.println("Testing neural-network...");
        time1 = System.currentTimeMillis();

        loadData("task_data_75000_test.csv", 1, SAMPLE_NUMBER, 512, LABEL_INDEX, CLASS_NUMBER);
        //loadData("random_task_data_1_1.csv", 1, 16450, 512, LABEL_INDEX, CLASS_NUMBER);
        //loadData("random_task_data_1_2.csv", 1, 14900, 512, LABEL_INDEX, CLASS_NUMBER);

        // create output for every test sample
        prediction = sMLNetwork.output(sTestData.getFeatureMatrix());
        System.out.println(prediction.toString());
        time2 = System.currentTimeMillis();
        System.out.println("sMLNetwork.output(): " + (time2 - time1));

        if (TEST_DATA_LABELLED == true) {
            // let evaluation print stats how often the right output had the highest value
            eval = new Evaluation(CLASS_NUMBER);
            eval.eval(sTestData.getLabels(), prediction);
            time3 = System.currentTimeMillis();
            System.out.println(eval.stats());
            System.out.println("eval.eval(): " + (time3 - time2));
        } else {
            mean = prediction.mean();
            System.out.println(mean.toString());
        }// if (TEST_DATA_LABELLED == true)

        System.out.println("Testing completed!");
    }

    private static void loadNetwork() {
        long time1, time2;

        System.out.println("Loading neural-network...");
        time1 = System.currentTimeMillis();

        sMLNetworkFile = new File(NN_MODEL_DIRECTORY + NN_MODEL_FILE);

        if (!sMLNetworkFile.exists()) {
            System.out.println("NN-Model not found");
            return;
        }// if (!sMLNetworkFile.exists())
        System.out.println("NN-Model found");

        try {
            sMLNetwork = ModelSerializer.restoreMultiLayerNetwork(sMLNetworkFile);
        } catch (IOException e) {
            System.out.println("LOADING - INTERRUPTED");
        }// try
        time2 = System.currentTimeMillis();

        System.out.println("ModelSerializer.restoreMultiLayerNetwork(): " + (time2 - time1));

        System.out.println("Loading completed!");
    }

    private static void saveNetwork() {
        long time1, time2;

        System.out.println("Saving neural-network...");
        time1 = System.currentTimeMillis();

        sMLNetworkFile = new File(NN_MODEL_DIRECTORY + NN_MODEL_FILE);

        try {
            ModelSerializer.writeModel(sMLNetwork, sMLNetworkFile, sSaveUpdater);
        } catch (IOException e) {
            System.out.println("SAVING - INTERRUPTED");
        }// try
        time2 = System.currentTimeMillis();

        System.out.println("ModelSerializer.writeModel(): " + (time2 - time1));

        System.out.println("Saving completed!");
    }

    private static void loadData(String fileName, int dataType, int sampleNumber, int batchSize, int labelIndex, int classNumber) {
        RecordReader recordReader;
        // number of lines-to-skip in csv-file (f.ex. skipping table-header)
        int numLinesToSkip = 0;
        // delimiter for row-values
        String delimiter = ";";

        // load and parse the csv-file with CSVRecordReader
        recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        try {
            recordReader.initialize(new FileSplit(new ClassPathResource(fileName).getFile()));
        } catch (IOException | InterruptedException e) {
            System.out.println("RECORD-READING - INTERRUPTED");
        }// try

        if (dataType == 0) {
            // convert the csv data with RecordReaderDataSetIterator to DataSet objects (ready for use in neural network)
            sTrainingDataIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, classNumber);

            // data normalization
            sDataNormalizer = new NormalizerMinMaxScaler(-1,1);
            // collect the statistics (min/max) from the training data
            sDataNormalizer.fit(sTrainingDataIterator);
            // set the normalization as pre-processor
            sTrainingDataIterator.setPreProcessor(sDataNormalizer);

            sTrainingData = sTrainingDataIterator.next(sampleNumber);

            // apply normalization to the training data
            sDataNormalizer.transform(sTrainingData);
        } else if (dataType == 1) {
            // convert the csv data with RecordReaderDataSetIterator to DataSet objects (ready for use in neural network)
            if (TEST_DATA_LABELLED == true) {
                sTestDataIterator = new RecordReaderDataSetIterator(recordReader, sampleNumber, labelIndex, classNumber);
            } else {
                sTestDataIterator = new RecordReaderDataSetIterator(recordReader, sampleNumber);
            }// if (TEST_DATA_LABELLED == true)

            sTestData = sTestDataIterator.next(sampleNumber);

            // apply normalization to the test data
            sDataNormalizer.transform(sTestData);
        }// if (dataType == 0)
    }

    private static void testMnistEarlyStopping() throws Exception {
        int nChannels = 1;
        int outputNum = 10;
        int batchSize = 25;
        int iterations = 1;
        int seed = 123;

        // configure network
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(true).l2(0.0005)
            .learningRate(0.02)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(20).dropOut(0.5)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, new DenseLayer.Builder()
                .nOut(500).build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            // see note in LenetMnistExample
            .setInputType(InputType.convolutionalFlat(28, 28, 1))
            .backprop(true).pretrain(false).build();

        // get mnist data
        DataSetIterator mnistTrain1024 = new MnistDataSetIterator(batchSize,1024,false,true,true,12345);
        DataSetIterator mnistTest512 = new MnistDataSetIterator(batchSize,512,false,false,true,12345);

        String tempDir = System.getProperty("java.io.tmpdir");
        String exampleDirectory = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/");
        // TODO: folder-creation necessary otherwise it results in a FileNotFound-Error
        File test = new File(exampleDirectory);
        test.mkdirs();
        System.out.println("exampleDirectory: " + exampleDirectory);

        EarlyStoppingModelSaver esSaver = new LocalFileModelSaver(exampleDirectory);
        EarlyStoppingConfiguration esConfiguration = new EarlyStoppingConfiguration.Builder()
            // max of 50 epochs
            .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
            .evaluateEveryNEpochs(1)
            // max of 20 minutes
            .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(5, TimeUnit.MINUTES))
            // calculate test set score
            .scoreCalculator(new DataSetLossCalculator(mnistTest512, true))
            .modelSaver(esSaver)
            .build();

        EarlyStoppingTrainer esTrainer = new EarlyStoppingTrainer(esConfiguration,configuration,mnistTrain1024);

        // conduct early stopping training:
        EarlyStoppingResult esResult = esTrainer.fit();
        System.out.println("Termination reason: " + esResult.getTerminationReason());
        System.out.println("Termination details: " + esResult.getTerminationDetails());
        System.out.println("Total epochs: " + esResult.getTotalEpochs());
        System.out.println("Best epoch number: " + esResult.getBestModelEpoch());
        System.out.println("Score at best epoch: " + esResult.getBestModelScore());

        // print score vs. epoch
        Map<Integer,Double> scoreVsEpoch = esResult.getScoreVsEpoch();
        List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Score vs. Epoch:");
        for (Integer i : list) {
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }// for (Integer i : list)
    }

    private static void testCSVEarlyStopping() throws Exception {
        String tempDir = System.getProperty("java.io.tmpdir");
        String exampleDirectory = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/");
        // TODO: folder-creation necessary otherwise it results in a FileNotFound-Error
        File test = new File(exampleDirectory);
        test.mkdirs();
        System.out.println("exampleDirectory: " + exampleDirectory);

        EarlyStoppingModelSaver esSaver = new LocalFileModelSaver(exampleDirectory);
        EarlyStoppingConfiguration esConfigurationiguration = new EarlyStoppingConfiguration.Builder()
            // max of 50 epochs
            .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
            .evaluateEveryNEpochs(1)
            // max of 20 minutes
            .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(5, TimeUnit.MINUTES))
            // calculate test set score
            .scoreCalculator(new DataSetLossCalculator(sTestDataIterator, false))
            .modelSaver(esSaver)
            .build();

        EarlyStoppingTrainer esTrainer = new EarlyStoppingTrainer(esConfigurationiguration,
            sMLNetworkConfiguration, sTrainingDataIterator);

        //Conduct early stopping training:
        EarlyStoppingResult esResult = esTrainer.fit();
        System.out.println("Termination reason: " + esResult.getTerminationReason());
        System.out.println("Termination details: " + esResult.getTerminationDetails());
        System.out.println("Total epochs: " + esResult.getTotalEpochs());
        System.out.println("Best epoch number: " + esResult.getBestModelEpoch());
        System.out.println("Score at best epoch: " + esResult.getBestModelScore());

        // print score vs. epoch
        Map<Integer,Double> scoreVsEpoch = esResult.getScoreVsEpoch();
        List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Score vs. Epoch:");
        for (Integer i : list) {
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }// for (Integer i : list)
    }

    /******************************************
     * MAIN
     *****************************************/
    public static void main(String[] args) {
        initializeNetwork();

        trainNetwork();

        testNetwork();

        try {
            testMnistEarlyStopping();
            testCSVEarlyStopping();
        } catch (Exception e) {
            System.out.println("EARLY-STOPPING-TEST - INTERRUPTED");
        }// try
    }
}
/*
 * *************************************************************************************
 * *************************************************************************************
 */
