function [imageUndistorted, undistortionTable] = undistortBarrel(imageDistorted)
%imageDistorted = imresize(imageDistorted, 2);

rowsImage = size(imageDistorted,1);
colsImage = size(imageDistorted,2);
undistortionTable = uint32(zeros((rowsImage*colsImage),3));

% We need to normalize our coordinate system!
[YRatio,XRatio] = getLinearLimits(rowsImage,colsImage);
stepsCols = linspace(-XRatio,XRatio,colsImage);
stepsRows = linspace(-YRatio,YRatio,rowsImage);
idxTable = 1;

k = -0.002;
for idxChannel=1:3
    for idxRows = 1:rowsImage
       for idxCols = 1:colsImage
            X1 = getXUndistorted(stepsCols(idxCols),stepsRows(idxRows),k);
            Y1 = getYUndistorted(stepsCols(idxCols),stepsRows(idxRows),k);

            % Now that we have our answer on the -1..1 space we need to convert
            % it back to the image coordinates
            X1 = findNearest(stepsCols,X1);
            Y1 = findNearest(stepsRows,Y1);
            imageUndistorted(idxRows,idxCols,idxChannel) = imageDistorted(Y1,X1,idxChannel);
            %undistortionTable(idxTable,[2 3]) = [Y1,X1];
            undistortionTable(idxTable,:) = [idxTable Y1,X1];
            idxTable = idxTable + 1;
       end
    end
end
%imageUndistorted = imresize(imageUndistorted, 0.5);
end

%% XUndistorted (Infered from mupad ....)
function [X1] = getXUndistorted(x_2,y_2,k_1)
%X1 = (x_2*((((250000.0*y_2^6)/(111.0*x_2^2 + 111.0*y_2^2)^2 + (37037037.037037037080153822898865*y_2^6)/(111.0*x_2^2 + 111.0*y_2^2)^3)^(1/2) + (500.0*y_2^3)/(111.0*x_2^2 + 111.0*y_2^2))^(1/3) - (333.33333333333333303727386009996*y_2^2)/((((250000.0*y_2^6)/(111.0*x_2^2 + 111.0*y_2^2)^2 + (37037037.037037037080153822898865*y_2^6)/(111.0*x_2^2 + 111.0*y_2^2)^3)^(1/2) + (500.0*y_2^3)/(111.0*x_2^2 + 111.0*y_2^2))^(1/3)*(111.0*x_2^2 + 111.0*y_2^2))))/y_2;
X1 = (x_2*(((25.0*y_2^3)/(11.0*x_2^2 + 11.0*y_2^2) + ((625.0*y_2^6)/(11.0*x_2^2 + 11.0*y_2^2)^2 + (4629.6296296296296333139252965339*y_2^6)/(11.0*x_2^2 + 11.0*y_2^2)^3)^(1/2))^(1/3) - (16.666666666666666685170383743753*y_2^2)/(((25.0*y_2^3)/(11.0*x_2^2 + 11.0*y_2^2) + ((625.0*y_2^6)/(11.0*x_2^2 + 11.0*y_2^2)^2 + (4629.6296296296296333139252965339*y_2^6)/(11.0*x_2^2 + 11.0*y_2^2)^3)^(1/2))^(1/3)*(11.0*x_2^2 + 11.0*y_2^2))))/y_2;
%X1 = x_2/y_2*((((1/4)*y_2^6/(k_1*x_2^2 + k_1*y_2^2)^2 +(1/27)*y_2^6/(k_1*x_2^2 + k_1*y_2^2)^3)^(1/2) + (1/2)*y_2^3/(k_1*x_2^2 + k_1*y_2^2))^(1/3) - (1/3)*y_2^2/(k_1*x_2^2 + k_1*y_2^2)/(((1/4)*y_2^6/(k_1*x_2^2 + k_1*y_2^2)^2 + (1/27)*y_2^6/(k_1*x_2^2 + k_1*y_2^2)^3)^(1/2) + (1\2)*y_2^3/(k_1*x_2^2 + k_1*y_2^2))^(1/3));
end

%% XUndistorted (Infered from mupad ....)
function [Y1] = getYUndistorted(x_2,y_2,k_1)
%Y1 = (((250000.0*y_2^6)/(111.0*x_2^2 + 111.0*y_2^2)^2 + (37037037.037037037080153822898865*y_2^6)/(111.0*x_2^2 + 111.0*y_2^2)^3)^(1/2) + (500.0*y_2^3)/(111.0*x_2^2 + 111.0*y_2^2))^(1/3) - (333.33333333333333303727386009996*y_2^2)/((((250000.0*y_2^6)/(111.0*x_2^2 + 111.0*y_2^2)^2 + (37037037.037037037080153822898865*y_2^6)/(111.0*x_2^2 + 111.0*y_2^2)^3)^(1/2) + (500.0*y_2^3)/(111.0*x_2^2 + 111.0*y_2^2))^(1/3)*(111.0*x_2^2 + 111.0*y_2^2));
Y1 = ((25.0*y_2^3)/(11.0*x_2^2 + 11.0*y_2^2) + ((625.0*y_2^6)/(11.0*x_2^2 + 11.0*y_2^2)^2 + (4629.6296296296296333139252965339*y_2^6)/(11.0*x_2^2 + 11.0*y_2^2)^3)^(1/2))^(1/3) - (16.666666666666666685170383743753*y_2^2)/(((25.0*y_2^3)/(11.0*x_2^2 + 11.0*y_2^2) + ((625.0*y_2^6)/(11.0*x_2^2 + 11.0*y_2^2)^2 + (4629.6296296296296333139252965339*y_2^6)/(11.0*x_2^2 + 11.0*y_2^2)^3)^(1/2))^(1/3)*(11.0*x_2^2 + 11.0*y_2^2));
%Y1 = (((1/4)*y_2^6/(k_1*x_2^2 + k_1*y_2^2)^2 + (1/27)*y_2^6/(k_1*x_2^2 + k_1*y_2^2)^3)^(1/2) + (1/2)*y_2^3/(k_1*x_2^2 + k_1*y_2^2))^(1/3) - (1/3)*y_2^2/(k_1*x_2^2 + k_1*y_2^2)/(((1/4)*y_2^6/(k_1*x_2^2 +k_1*y_2^2)^2 + (1/27)*y_2^6/(k_1*x_2^2 + k_1*y_2^2)^3)^(1/2) + (1/2)*y_2^3/(k_1*x_2^2 + k_1*y_2^2))^(1/3);
end

%% Get aspect ration of the image to calculate the linear space
function [YRatio, XRatio] = getLinearLimits(nRows, nCols)
YRatio = nRows*(1/(nCols^2 + nRows^2))^(1/2);
XRatio = nCols*(1/(nCols^2 + nRows^2))^(1/2);
end

%% Find index of nearest value
function [idxNear] = findNearest(vector, val)
    tmp = abs(vector - val);
    [~, idxNear] = min(tmp) ;
end