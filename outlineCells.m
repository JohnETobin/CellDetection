function [] = outlineCells(inputImagesFolder, outputImagesPath, network)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

IMAGE_DIMENSION = 512;
EROSION_RADIUS = 2;


load(network);
%net = load("net.mat");
imagesPathDir = dir(inputImagesFolder);
imagesPathDir([imagesPathDir.isdir]) = [];
numImages = length(imagesPathDir);

for i = 1:numImages
    currFile = fullfile(inputImagesFolder, imagesPathDir(i).name);
    currImg = im2double(rgb2gray(imread(currFile)));
    currImg = imresize(currImg, [IMAGE_DIMENSION IMAGE_DIMENSION]);
    originalCellImagesArr{i} = currImg;
    currImg = ((currImg - mean(currImg(:)))/var(currImg(:)));
    cellImagesArr{i} = currImg;
end

for i = 1:numImages
    result = semanticseg(cellImagesArr{i}, fullImageViewNet);
    
    binaryResult = zeros(128, 128);
    binaryResult(result == "cellCenter") = 1;
    
    upsizedResult = round(imresize(binaryResult, [IMAGE_DIMENSION IMAGE_DIMENSION]));
    
    SE = strel('disk', EROSION_RADIUS);
    erodedResult = imerode(upsizedResult, SE);
    
    colorCellImg = zeros(IMAGE_DIMENSION, IMAGE_DIMENSION, 3);
    
    for j = 1 : 3
    colorCellImg(:,:,j) = originalCellImagesArr{i};
    end
    
    cellOutlines = zeros(IMAGE_DIMENSION, IMAGE_DIMENSION);
    cellOutlines(upsizedResult == 1 & erodedResult == 0) = 1;
    
    layer1 = colorCellImg(:, :, 1);
    layer1(cellOutlines == 1) = 256;
    layer2 = colorCellImg(:, :, 2);
    layer2(cellOutlines == 1) = 0;
    layer3 = colorCellImg(:, :, 3);
    layer3(cellOutlines == 1) = 0;
    
    colorCellImg(:,:,1) = layer1;
    colorCellImg(:,:,2) = layer2;
    colorCellImg(:,:,3) = layer3;
    
    binaryCellImg = sprintf('BinaryCellImage%d.png', i);
    binaryCellImgPath = fullfile(outputImagesPath, binaryCellImg);
    imwrite(upsizedResult,binaryCellImgPath,'png');
    
    labeledCellImg = sprintf('LabeledCellImage%d.png', i);
    labeledCellImgPath = fullfile(outputImagesPath, labeledCellImg);
    imwrite(colorCellImg,labeledCellImgPath,'png');
end

figure;
imshow(upsizedResult, []);

figure;
imshow(erodedResult, []);

figure;
imshow(cellOutlines, []);
