
Train = imageDatastore("dataset\train\", 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
Validation = imageDatastore("dataset\validation\", 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
Test = imageDatastore("dataset\test\", 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
inputSize = [224 224];
Train.ReadFcn = @(loc)imresize(imread(loc),inputSize);
Validation.ReadFcn = @(loc)imresize(imread(loc),inputSize);
Test.ReadFcn = @(loc)imresize(imread(loc),inputSize);
tbl = countEachLabel(Train); %count number of images in dataset

imageAugmenter = imageDataAugmenter('RandRotation', [-20,20], 'RandXTranslation', [-3,3], 'RandYTranslation', [-3,3])
imageSize = [224 224 3];
augimds = augmentedImageDatastore(imageSize, Train, 'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb')

layers = [
    imageInputLayer(imageSize) %input layer (normalization possible to add)
    
    convolution2dLayer(3,4,'Padding','same') %convolution layer with sliding filters (8 filters 3x3) (Padding - padding of the input edge - same size in and out)
    %option with adding parameters for weights / deviations
    batchNormalizationLayer %normalizing layer for mini batches. It speeds up learning and reduces sensitivity
    reluLayer %threshold operation. value less than zero is set to 0
    
    maxPooling2dLayer(2,'Stride',2) %downsampling. splits the input data into rectangles and computes the maximum of each (pool size [2 2] with step [3 3]
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(height(tbl)) %multiplies the input data by the weight matrix and adds the deviation vector
    softmaxLayer %softmax activation function - changes real values ​​into probabilities
    classificationLayer]; %calculates entropy loss

options = trainingOptions('sgdm', 'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor' ,0.8 , 'LearnRateDropPeriod', 5, 'ValidationData', Validation, 'MaxEpochs', 100, 'MiniBatchSize', 64, 'Plots', 'training-progress', 'Shuffle', 'once','ValidationPatience',3, 'OutputNetwork','last-iteration')
%LearnRateDropFactor and LearnRateDropPeriod
net = trainNetwork(augimds,layers,options)

TestPred = classify(net, Test);
PredTest = Test.Labels;

accuracyTest = sum(TestPred == PredTest)/numel(PredTest)

[C1, order1] = confusionmat(Test.Labels, TestPred)
figure, confusionchart(C1)
title('Matrix of test set errors')


ValidationPred = classify(net, Validation);
PredValidation = Validation.Labels;

accuracyValidation = sum(ValidationPred == PredValidation)/numel(PredValidation)

[C2, order2] = confusionmat(Validation.Labels, ValidationPred)
figure, confusionchart(C2)
title('Matrix of validation set errors')


TrainPred = classify(net, Train);
PredTrain = Train.Labels;

accuracy = sum(TrainPred == PredTrain)/numel(PredTrain)

[C3, order3] = confusionmat(Train.Labels, TrainPred)
figure, confusionchart(C3)
title('Matrix of train set errors')


save('cnn.mat', 'net')