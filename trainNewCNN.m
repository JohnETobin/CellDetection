function [] = trainNewCNN(trainingImagesPath)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%setup('useGpu', true);

IMAGE_DIMENSION = 512;
CELL_RADIUS = 18;
CELL_CENTER_RADIUS = 1;
DILATION_RADIUS = 2;
CURRENT_DIRECTORY = pwd;

imagesPathDir = dir(trainingImagesPath);
imagesPathDir([imagesPathDir.isdir]) = [];
numImages = length(imagesPathDir);

for i = 1:numImages
    currFile = fullfile(trainingImagesPath, imagesPathDir(i).name);
    currImg = im2double(rgb2gray(imread(currFile)));
    currImg = imresize(currImg, [IMAGE_DIMENSION IMAGE_DIMENSION]);
    originalCellImagesArr{i} = currImg;
    currImg = ((currImg - mean(currImg(:)))/var(currImg(:)));
    cellImagesArr{i} = currImg;
end

[IoIDownsampledWiderCellCenterIndices, IoIDownsampledCellCenterIndices, ... 
    IoIDownsampledWiderNaNIndices, IoIDownsampledNaNIndices] = ...
    generateDownsampledTrainingData(cellImagesArr, IMAGE_DIMENSION, CELL_RADIUS, CELL_CENTER_RADIUS, DILATION_RADIUS);

%CREATE IoIDatastore AND groundTruthDatastore BEFOREHAND MANUALLY

IoIDSPath = append(CURRENT_DIRECTORY, '/IoIDatastore');
groundTruthDSPath = append(CURRENT_DIRECTORY, '/groundTruthDatastore');

for i = 1:size(cellImagesArr, 2)
   %imwrite(imagesOfInterest{i}(1:4:end, 1:4:end), append(CURRENT_DIRECTORY, '/IoIDatastore/', ...
   %    'cellsImg', num2str(i), '.png')); 
   imwrite(cellImagesArr{i}, append(CURRENT_DIRECTORY, '/IoIDatastore/', ...
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
customFocalLoss = focalLossLayer('Name', 'customFocalLoss', 'Alpha', 0.1, 'Gamma', 5);

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
    convolution2dLayer(4, 128, 'Padding', 'same') % Mimics fully connected layer
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

newCNN = fullImageViewNet;

save('newCNN.mat', 'newCNN');
end

