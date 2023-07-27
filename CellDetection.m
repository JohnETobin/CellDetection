IMAGE_DIMENSION = 512;

CELL_CENTER_RADIUS = 1;

CELL_RADIUS = 18;

DILATION_RADIUS = 2;

NAN_BOUNDARY = 2;

CURRENT_DIRECTORY = pwd;
% Open images from folders and save to array

%cellPhotos1 = dir(append(CURRENT_DIRECTORY, '/cellPhotos1/'));
cellPhotos2 = dir(append(CURRENT_DIRECTORY, '/cellPhotos2/'));
cellPhotos3 = dir(append(CURRENT_DIRECTORY, '/cellPhotos3/'));
cellPhotos4 = dir(append(CURRENT_DIRECTORY, '/cellPhotos4/'));

%cellPhotos1 = cellPhotos1(3:length(cellPhotos1));
cellPhotos2 = cellPhotos2(3:length(cellPhotos2));
cellPhotos3 = cellPhotos3(3:length(cellPhotos3));
cellPhotos4 = cellPhotos4(3:length(cellPhotos4));

%numCellPhotos1 = length(cellPhotos1);
numCellPhotos2 = length(cellPhotos2);
numCellPhotos3 = length(cellPhotos3);
numCellPhotos4 = length(cellPhotos4);

% for i = 1:numCellPhotos1
%     currFile = "cellPhotos1/" + cellPhotos1(i).name;
%     currImg = im2double(rgb2gray(imread(currFile)));
%     currImg = imresize(currImg, [IMAGE_DIMENSION IMAGE_DIMENSION]);
%     currImg = ((currImg - mean(currImg(:)))/var(currImg(:)));
%     cellPhotos1Arr{i} = currImg;
% end

for i = 1:numCellPhotos2
    currFile = "cellPhotos2/" + cellPhotos2(i).name;
    currImg = im2double(rgb2gray(imread(currFile)));
    currImg = imresize(currImg, [IMAGE_DIMENSION IMAGE_DIMENSION]);
    currImg = ((currImg - mean(currImg(:)))/var(currImg(:)));
    cellPhotos2Arr{i} = currImg;
end

for i = 1:numCellPhotos3
    currFile = "cellPhotos3/" + cellPhotos3(i).name;
    currImg = im2double(rgb2gray(imread(currFile)));
    currImg = imresize(currImg, [IMAGE_DIMENSION IMAGE_DIMENSION]);
    currImg = ((currImg - mean(currImg(:)))/var(currImg(:)));
    cellPhotos3Arr{i} = currImg;
end

for i = 1:numCellPhotos4
    currFile = "cellPhotos4/" + cellPhotos4(i).name;
    currImg = im2double(rgb2gray(imread(currFile)));
    currImg = imresize(currImg, [IMAGE_DIMENSION IMAGE_DIMENSION]);
    currImg = ((currImg - mean(currImg(:)))/var(currImg(:)));
    cellPhotos4Arr{i} = currImg;
end

% figure;
% imshow(cellPhotos2Arr{1}, []);
% figure;
% imshow(cellPhotos3Arr{1}, []);
% figure;
% imshow(cellPhotos4Arr{1}, []);

%DONE 0: Load images and make them the same size 512 x 512
%DONE 0: Subtract mean and divide by variance for each image
%% Benchmark 1
%TODO 1: Get ground truth values from 3 images using ginput()

% Select images we will be assigning ground truth values to
imagesOfInterest = {cellPhotos2Arr{1}, cellPhotos3Arr{1}, cellPhotos4Arr{1}};
% Make the tensor that will contain the single pixel cell center indices
IoICellCenterIndices = zeros(IMAGE_DIMENSION, IMAGE_DIMENSION, size(imagesOfInterest, 1));
% Make the tensor that will contain the multi-pixel, wider cell center indices
IoIWiderCellCenterIndices = IoICellCenterIndices;

% Uncomment if you want to generate new training data
%[IoIWiderCellCenterIndices, IoICellCenterIndices] = generateTrainingData(imagesOfInterest);

load trainingDataIndices.mat;

% cellPhotos and nonCellPhotos will hold the training image data for the NN
% dimensions of the training images are (2*CELL_RADIUS) by (2*CELL_RADIUS)
trainingCellPhotos = {};
trainingNonCellPhotos = {};
cellPhotosCounter = 1;
nonCellPhotosCounter = 1;

indices = find(IoIWiderCellCenterIndices == 1);
numCells = size(indices, 1);
nonCellSamplingFrequency = round(IMAGE_DIMENSION^2 / numCells);
sampleCounter = 0;

for i = 1:size(imagesOfInterest, 2) % image slice index
    currImage = imagesOfInterest{i};
    for j = CELL_RADIUS : (IMAGE_DIMENSION - CELL_RADIUS) % adjust row index
        for k = CELL_RADIUS : (IMAGE_DIMENSION - CELL_RADIUS) % adjust column index
            sampleCounter = sampleCounter + 1;
            if IoIWiderCellCenterIndices(j, k, i) == 1
                % assign the cell photo to the cellPhotos tensor
                trainingCellPhotos{cellPhotosCounter} = currImage(j-(CELL_RADIUS-1) : j+CELL_RADIUS, k-(CELL_RADIUS-1) : k+CELL_RADIUS);
                cellPhotosCounter = cellPhotosCounter + 1;
            elseif sampleCounter == nonCellSamplingFrequency
                trainingNonCellPhotos{nonCellPhotosCounter} = currImage(j-(CELL_RADIUS-1) : j+CELL_RADIUS, k-(CELL_RADIUS-1) : k+CELL_RADIUS);
                nonCellPhotosCounter = nonCellPhotosCounter + 1;
                sampleCounter = 0;
            end
            
        end
    end
end

% TODO 2: Learn the matlab CNN toolkit
% We referenced the Math Works tutorial on CNNs
    % https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
% We also referenced the letter characterization CNN from the Convolutional Neural Networks Practical
    % authored by Andrea Vedaldi and Andrew Zisserman (Release 2017a), with edits by Jerod Weinman.

%%    
% TODO 3: Make a simple 1-convolution layer and sigmoidal layer CNN
cellDSPath = append(CURRENT_DIRECTORY, '/firstBenchmarkCellDataset');
cellPositiveImgs = dir(append(CURRENT_DIRECTORY, '/firstBenchmarkCellDataset', '/cellPositiveImgs'));
cellNegativeImgs = dir(append(CURRENT_DIRECTORY, '/firstBenchmarkCellDataset', '/cellNegativeImgs'));

for i = 1:size(trainingCellPhotos, 2)
   imwrite(trainingCellPhotos{i}, append(CURRENT_DIRECTORY, '/firstBenchmarkCellDataset', ...
       '/cellPositiveImgs/', 'cell', num2str(i), '.png')); 
end

for i = 1:(floor(size(trainingNonCellPhotos, 2) / size(trainingCellPhotos, 2))):size(trainingNonCellPhotos, 2)
   imwrite(trainingNonCellPhotos{i}, append(CURRENT_DIRECTORY, '/firstBenchmarkCellDataset', ...
       '/cellNegativeImgs/', 'cell', num2str(i), '.png')); 
end

% %%
% setup('useGpu', true);
% 
% numberOfTrainingImgs = 1200;
% widthOfConvKernel = 8;
% numberOfFilters = 8;
% 
% % code below taken Math Works tutorial on CNNs © 1994-2022 The MathWorks, Inc.
% % https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
% cellsDS = imageDatastore(cellDSPath,'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% % split the data into training and testing images
% [cellDSTrain,cellDSTest] = splitEachLabel(cellsDS,numberOfTrainingImgs,'randomize');
% 
% % **create basic CNN architecture**
% CNNLayers = [
%     imageInputLayer([(CELL_RADIUS*2) (CELL_RADIUS*2) 1]) % here we set the size of the input images
%     
%     convolution2dLayer(widthOfConvKernel, numberOfFilters) % here we set the size of the convolution kernel and filter count
%     batchNormalizationLayer
%     sigmoidLayer
%     
%     fullyConnectedLayer(2) % here we specify that there are two possible classes for each image: cell or not cell
%     softmaxLayer
%     classificationLayer
% ];
% 
% % **train the CNN**
% % code below taken from Math Works tutorial on CNNs © 1994-2022 The MathWorks, Inc.
% % https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
% CNNOptions = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.01, ...
%     'MaxEpochs',20, ... %INCREASE EPOCH???
%     'Shuffle','every-epoch', ...
%     'ValidationData',cellDSTest, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% % basicNet = trainNetwork(cellDSTrain, CNNLayers, CNNOptions);
% % save('basicNeuralNet.mat', 'basicNet');
%%
load basicNeuralNet

%% Single Cell CNN
% Add layers

setup('useGpu', true);

numberOfTrainingImgs = 1200;
widthOfConvKernel = 4; %EXPERIMENT WITH THIS NUMBER
numberOfFilters = 16; %EXPERIMENT WITH THIS NUMBER

% code below taken Math Works tutorial on CNNs © 1994-2022 The MathWorks, Inc.
% https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
cellsDS = imageDatastore(cellDSPath,'IncludeSubfolders',true,'LabelSource','foldernames');


% split the data into training and testing images
[cellDSTrain,cellDSTest] = splitEachLabel(cellsDS,numberOfTrainingImgs,'randomize');


% **create new CNN architecture with conv layer instead of fully connected at end**

CNN2Layers = [
    imageInputLayer([(CELL_RADIUS*2) (CELL_RADIUS*2) 1]) % here we set the size of the input images
   
    convolution2dLayer(4, 16) % here we set the size of the convolution kernel and filter count
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(4, 16) % here we set the size of the convolution kernel and filter count
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4, 32) % here we set the size of the convolution kernel and filter count
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(2) % here we specify that there are two possible classes for each image: cell or not cell
    softmaxLayer
    classificationLayer
];



% **train the CNN**
% code below taken from Math Works tutorial on CNNs © 1994-2022 The MathWorks, Inc.
% https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
CNN2Options = trainingOptions('adam', ... %adam is better
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ... %INCREASE EPOCH???
    'Shuffle','every-epoch', ...
    'ValidationData',cellDSTest, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');



cpcpNet = trainNetwork(cellDSTrain, CNN2Layers, CNN2Options);



save('cpcpNeuralNet.mat', 'cpcpNet');

%% Multi Cell CNN
% Make the CNN more accurate and make it take in full images at a time

%TODO 1: Update how we assign ground truth (i.e. get a 4x downsampled binary
%      ground truth matrix for each image with ginput())
%TODO 2: Create the randomPatchExtractionDatastore for trainNetwork
%TODO 3: Create 'options' for trainNetwork
%TODO 4: MAYBE try to get classify to work on a whole image

setup('useGpu', true);

% Select images we will be assigning ground truth values to
imagesOfInterest = {cellPhotos2Arr{1}, cellPhotos3Arr{1}, cellPhotos4Arr{1},...
    cellPhotos2Arr{2}, cellPhotos3Arr{2}, cellPhotos4Arr{2}, ...
    cellPhotos2Arr{3}, cellPhotos3Arr{3}, cellPhotos4Arr{3}, ...
    cellPhotos2Arr{4}, cellPhotos3Arr{4}, cellPhotos4Arr{4}, ...
    cellPhotos2Arr{5}, cellPhotos3Arr{5}, cellPhotos4Arr{5}};

%Uncomment to assign ground truth
%  [IoIDownsampledWiderCellCenterIndices, IoIDownsampledCellCenterIndices, IoIDownsampledWiderNaNIndices, IoIDownsampledNaNIndices] = ...
%      generateDownsampledTrainingData(imagesOfInterest, IMAGE_DIMENSION, CELL_RADIUS, CELL_CENTER_RADIUS, DILATION_RADIUS);

%load('downsampledIndices.mat')
%load('cellsNaNsBackgroundData.mat');
%load('cellsNaNsBackgroundData9Photos.mat');
load('15CellPhotosTraining.mat');

%CREATE IoIDatastore AND groundTruthDatastore BEFOREHAND MANUALLY

IoIDSPath = append(CURRENT_DIRECTORY, '/IoIDatastore');
groundTruthDSPath = append(CURRENT_DIRECTORY, '/groundTruthDatastore');

for i = 1:size(imagesOfInterest, 2)
   %imwrite(imagesOfInterest{i}(1:4:end, 1:4:end), append(CURRENT_DIRECTORY, '/IoIDatastore/', ...
   %    'cellsImg', num2str(i), '.png')); 
   imwrite(imagesOfInterest{i}, append(CURRENT_DIRECTORY, '/IoIDatastore/', ...
       'cellsImg', num2str(i), '.png')); 
   %USED IoIDownsampledWider HERE, may need to change
%    imwrite(IoIDownsampledWiderCellCenterIndices(:,:,i), append(CURRENT_DIRECTORY, '/groundTruthDatastore/', ...
%        'cellsGTImg', num2str(i), '.png'));
    imwrite(IoIDownsampledWiderNaNIndices(:,:,i), append(CURRENT_DIRECTORY, '/groundTruthDatastore/', ...
       'cellsGTImg', num2str(i), '.png'));

end

IoIDatastore = imageDatastore(IoIDSPath);
classNames = ["cellCenter", "background"];
labelIDs = [1 0];
groundTruthDatastore = pixelLabelDatastore(groundTruthDSPath, classNames, labelIDs);%imageDatastore(groundTruthDSPath);


%PATCHSIZE = (CELL_RADIUS / 4) * 2;
% code below taken Math Works randomPatchExtractionDatastore
%fullImgTrainingDS = randomPatchExtractionDatastore(IoIDatastore, groundTruthDatastore, [PATCHSIZE PATCHSIZE]);
fullImgTrainingDS = combine(IoIDatastore, groundTruthDatastore);

% split the data into training and testing images
%numberOfTrainingImgs = 2;
%[rPEDTrain,rPEDTest] = splitEachLabel(fullImgTrainingDS,numberOfTrainingImgs,'randomize');

%customFocalLoss = focalLossLayer('Name', 'customFocalLoss', 'Alpha', 0.1, 'Gamma', 4.3);
customFocalLoss = focalLossLayer('Name', 'customFocalLoss', 'Alpha', 0.1, 'Gamma', 4);

CNN3Layers = [
    %imageInputLayer([IMAGE_DIMENSION-CELL_RADIUS*2 IMAGE_DIMENSION-CELL_RADIUS*2 1]) % here we set the size of the input images
    imageInputLayer([IMAGE_DIMENSION IMAGE_DIMENSION 1])
   
    convolution2dLayer(3, 48, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 48, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3, 64,'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 64,'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    %MODIFY THE KERNEL SIZE HERE
    convolution2dLayer(3, 128, 'Padding', 'same') % Mimics fully connected layer
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(1, 256, 'Padding', 'same') % Mimics fully connected layer
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(1, 2, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    softmaxLayer
    customFocalLoss
    %pixelClassificationLayer
    %focalLossLayer
];



% **train the CNN**
% code below taken from Math Works tutorial on CNNs © 1994-2022 The MathWorks, Inc.
% https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
% CNN3Options = trainingOptions('adam', ... %adam is better
%     'InitialLearnRate',0.01, ...
%     'MaxEpochs',5, ... %INCREASE EPOCH???
%     'Shuffle','every-epoch', ... %'ValidationData',fullImgTrainingDS, ... %FIX ME!!!!! %'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');

%Code below retrieved from MATLAB tutorial: 
%   https://www.mathworks.com/help/deeplearning/ug/define-custom-pixel-classification-layer-with-tversky-loss.html
CNN3Options = trainingOptions('adam', ...
    'GradientDecayFactor', 0.9, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate',0.005, ... %0.001
    'MaxEpochs',50, ...
    'LearnRateDropFactor',0.9, ...
    'LearnRateDropPeriod',10, ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',1); %was 50

fullImageViewNet = trainNetwork(fullImgTrainingDS, CNN3Layers, CNN3Options);
%fullImageViewNet = trainNetwork(groundTruthDatastore, CNN3Layers, CNN3Options);


save('fullImageViewNet.mat', 'fullImageViewNet');

testClassification(cellPhotos2Arr, fullImageViewNet);


%%

testimg = cellPhotos3Arr{1};

shrunkimg = testimg(1:4:end,1:4:end);

figure;
imshow(shrunkimg, [])




%% Benchmark 3


%% Benchmark 4


%% CNN Graveyard
% CNN3Layers = [
%     %imageInputLayer([IMAGE_DIMENSION-CELL_RADIUS*2 IMAGE_DIMENSION-CELL_RADIUS*2 1]) % here we set the size of the input images
%     imageInputLayer([IMAGE_DIMENSION IMAGE_DIMENSION 1])
%    
%     convolution2dLayer(8, 16, 'Padding', 'same')
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(8, 32,'Padding', 'same')
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     
%     %MODIFY THE KERNEL SIZE HERE
%     convolution2dLayer(8, 64, 'Padding', 'same') % Mimics fully connected layer
%     batchNormalizationLayer
%     reluLayer
%     
%     convolution2dLayer(1, 128, 'Padding', 'same') % Mimics fully connected layer
%     batchNormalizationLayer
%     reluLayer
%     
%     convolution2dLayer(1, 2, 'Padding', 'same')
%     batchNormalizationLayer
%     reluLayer
%     
%     softmaxLayer
%     pixelClassificationLayer
% ];