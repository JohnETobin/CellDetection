function [IoIDownsampledWiderCellCenterIndices, IoIDownsampledCellCenterIndices,IoIDownsampledWiderNaNIndices, IoIDownsampledNaNIndices] = ...
    generateDownsampledTrainingData(imagesOfInterest, IMAGE_DIMENSION, CELL_RADIUS, CELL_CENTER_RADIUS, DILATION_RADIUS)
% Get the single pixel and multipixel cell centers for each image of interest

% Make the tensor that will contain the single pixel cell center indices
IoIDownsampledCellCenterIndices = uint8(zeros(IMAGE_DIMENSION/4, IMAGE_DIMENSION/4, size(imagesOfInterest, 1)));
% Make the tensor that will contain the multi-pixel, wider cell center indices
IoIDownsampledWiderCellCenterIndices = IoIDownsampledCellCenterIndices;
IoIDownsampledNaNIndices = IoIDownsampledCellCenterIndices;
IoIDownsampledWiderNaNIndices = IoIDownsampledCellCenterIndices;


for i = 1:size(imagesOfInterest, 2)
    figure('name', 'Training Data Selection');
    imshow(imagesOfInterest{i}, []);
    hold on;
    title("Select the Center of Cells in this Image");
    line([1 IMAGE_DIMENSION], [4 4], 'Color', 'r');
    line([1 IMAGE_DIMENSION], [IMAGE_DIMENSION - 4 IMAGE_DIMENSION - 4], 'Color', 'r');
    line([4 4], [1 IMAGE_DIMENSION], 'Color', 'r');
    line([IMAGE_DIMENSION - 4 IMAGE_DIMENSION - 4], [1 IMAGE_DIMENSION] , 'Color', 'r');
    %line([1 IMAGE_DIMENSION], [CELL_RADIUS CELL_RADIUS], 'Color', 'r');
    %line([1 IMAGE_DIMENSION], [IMAGE_DIMENSION - CELL_RADIUS IMAGE_DIMENSION - CELL_RADIUS], 'Color', 'r');
    %line([CELL_RADIUS CELL_RADIUS], [1 IMAGE_DIMENSION], 'Color', 'r');
    %line([IMAGE_DIMENSION - CELL_RADIUS IMAGE_DIMENSION - CELL_RADIUS], [1 IMAGE_DIMENSION] , 'Color', 'r');
    while 1
      [yy, xx, button] = ginput(1);
      xx = round(xx);
      yy = round(yy);
      plot(yy, xx, 'r+');
      if button == 27 % user presses escape
            break
      end
      xx = ceil(xx/4);
      yy = ceil(yy/4);
        % assign the selected point to be a 1 in the indices tensor
        IoIDownsampledCellCenterIndices(xx, yy, i) = 1;
        IoIDownsampledNaNIndices(xx, yy, i) = 1;
        % assign the box pixel region around the selected point to be 1
        IoIDownsampledWiderCellCenterIndices((xx-ceil((CELL_CENTER_RADIUS/4))):(xx+ceil(CELL_CENTER_RADIUS/4)),...
            (yy-ceil(CELL_CENTER_RADIUS/4)):(yy+ceil(CELL_CENTER_RADIUS/4)), i) = 1;
        IoIDownsampledWiderNaNIndices((xx-ceil((CELL_CENTER_RADIUS/4))):(xx+ceil(CELL_CENTER_RADIUS/4)),...
            (yy-ceil(CELL_CENTER_RADIUS/4)):(yy+ceil(CELL_CENTER_RADIUS/4)), i) = 1;
    end
    
    SE = strel('disk', DILATION_RADIUS);
    IoIDownsampledNaNIndices(:,:,i) = imdilate(IoIDownsampledNaNIndices(:,:,i), SE);
    IoIDownsampledWiderNaNIndices(:,:,i) = imdilate(IoIDownsampledWiderNaNIndices(:,:,i), SE);
    
    temp1 = IoIDownsampledNaNIndices(:,:,i);
    temp2 = IoIDownsampledWiderNaNIndices(:,:,i);
    
    temp1(IoIDownsampledNaNIndices(:,:,i) == 1) = 2;
    temp2(IoIDownsampledWiderNaNIndices(:,:,i) == 1) = 2;
    
    temp1(IoIDownsampledCellCenterIndices(:,:,i) == 1) = 1;
    temp2(IoIDownsampledWiderCellCenterIndices(:,:,i) == 1) = 1;
    
    IoIDownsampledNaNIndices(:,:,i) = temp1;
    IoIDownsampledWiderNaNIndices(:,:,i) = temp2;
    
    close('name', 'Training Data Selection');
end

end
