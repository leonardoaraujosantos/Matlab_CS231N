classdef Transforms < handle
    %AUGMENTIMAGE Class used to create augmented images for trainning
    
    properties
    end
    
    methods
        function [imgCellOut] = doAugmentation(obj, imageIn)
            
        end
        
        function [grayImg] = convertToGrayscale(obj,imgIn)
            % Select channels
            R = imgIn(:,:,1);
            G = imgIn(:,:,2);
            B = imgIn(:,:,3);
            % Convert to grayscale but keep the number of channels.
            grayImg(:,:,1) = 0.2989 * R + 0.5870 * G + 0.1140 * B;
            grayImg(:,:,2) = grayImg(:,:,1);
            grayImg(:,:,3) = grayImg(:,:,1);
        end
        
        function [imgSepia] = sepiaFilter(obj, imageIn)
            % Select channels
            R = imageIn(:,:,1);
            G = imageIn(:,:,2);
            B = imageIn(:,:,3);
            % Convert to grayscale but keep the number of channels.
            imgSepia(:,:,1) = 0.393 * R + 0.769 * G + 0.189 * B;
            imgSepia(:,:,2) = 0.349 * R + 0.686 * G + 0.168 * B;
            imgSepia(:,:,3) = 0.272 * R + 0.534 * G + 0.131 * B;
        end
        
        function [flipImg] = flip_H_Image(obj,imageIn, prob)
            if rand() < prob
                flipImg = flip(imageIn,2);
            else
                flipImg = imageIn;
            end
        end
        
        function [flipImg] = flip_V_Image(obj,imageIn)
            flipImg = flip(imageIn,1);
        end
        
        function [flipImg] = flip_Color_Image(obj,imageIn)
            flipImg = flip(imageIn,3);
        end
        
        function [cropImg] = randomCrop(obj, imageIn)
            [nrows,ncols, ~] = size(imageIn);
            
            % Standard (Alexnet-paper) ratio for crop
            cropSizeRows = nrows * 0.875;
            cropSizeCols = ncols * 0.875;
            
            centerCropRow = (nrows-cropSizeRows)/2;
            centerCropCol = (ncols - cropSizeCols)/2;
            
            % We're going to use imcrop which parameter is a rect with
            % format: [xmin ymin width height], here width=cropSizeCols adn
            % height=cropSizeRows.
            % Return the center crop + (random numCrops-1)
            centerImage = imcrop(imageIn, [centerCropCol centerCropRow cropSizeCols-1 cropSizeRows-1]);
            
            cropImg = zeros([cropSizeRows,cropSizeCols,3,11]);
            cropImg(:,:,:,1) = centerImage;
            
            % Get 10 random crops excluding the center crop
            nImages = 1;
            while nImages < 10
                randX = randi(ncols - cropSizeCols);
                randY = randi(nrows - cropSizeRows);
                
                if (randX == centerCropCol) &&  (randY == centerCropCol)
                    continue;
                else
                    nImages = nImages + 1;
                    img = imcrop(imageIn, [randX randY cropSizeCols-1 cropSizeRows-1]);
                    cropImg(:,:,:,nImages) = img;
                end
            end
        end
        
        % Color normalize must be done to every image before
        % training/test/prediction
        function [normalizedImg] = colorNormalize(obj, imageIn)
            R = imageIn(:,:,1);
            G = imageIn(:,:,2);
            B = imageIn(:,:,3);
            
            % Get mean
            R_mean = mean2(R);
            G_mean = mean2(G);
            B_mean = mean2(B);
            
            % Get standard deviation
            R_std = std2(R);
            G_std = std2(G);
            B_std = std2(B);
            
            % For each channel subtract mean and divide by standard
            % deviation
            normalizedImg(:,:,1) = (double(R) - R_mean) / R_std;
            normalizedImg(:,:,2) = (double(G) - G_mean) / G_std;
            normalizedImg(:,:,3) = (double(B) - B_mean) / B_std;
        end
        
        function [chImg] = changeLumination(obj, imageIn)
            % The cie lab color space has better control for illumination
            % than HSL/HSV (But slower to compute)
            lab_image = rgb2lab(imageIn);
            
            channel = 1;
            
            meanValue = mean2(lab_image(:,:,channel));            
            
            % Change value for illumination            
            lab_image(:,:,channel) = lab_image(:,:,channel) + randi([int32(-meanValue),int32(meanValue)]);
            
            chImg = lab2rgb(lab_image);
        end
        
        function [colorJitImg] = colorJittering(obj, imageIn)
            % Channel to jitter (one at a time, or all of them)
            channel = randi([1 4]);
            
            % Use a random value bigger than 0.4 but less then 1
            randVal = 0.4 + rand();
            if randVal > 1
                randVal = 1;
            end
            
            randVal_R = 0.4 + rand(); randVal_R(randVal_R>1)=1;
            randVal_G = 0.4 + rand(); randVal_G(randVal_G>1)=1;
            randVal_B = 0.4 + rand(); randVal_B(randVal_B>1)=1;
            switch channel
                case 1
                    colorJitImg = imadjust(imageIn,[0 0 0; randVal 1 1],[]);
                case 2
                    colorJitImg = imadjust(imageIn,[0 0 0; 1 randVal 1],[]);
                case 3
                    colorJitImg = imadjust(imageIn,[0 0 0; 1 1 randVal],[]);
                case 4
                    colorJitImg = imadjust(imageIn,[0 0 0; randVal_R randVal_G randVal_B],[]);
            end
        end
        
        function [rotImages] = addRotation(obj, imageIn)
            rotImages = imrotate(imageIn,randi([-8 8]),'crop');
        end
        
        function [noiseImg] = addPeperNoise(obj, imageIn)
            noiseImg = imnoise(imageIn,'gaussian', 0, 0.01 * rand());
        end
        
        % Get the eigenVector and eigenValue of the covariance matrix
        % to implement the Alexnet Pca color augmentation.
        function [eigVec, eigVal] = getPcaBatch(obj, imgBatch)
            % Get a vector of each R,G,B channels from the batch
            R_batch = imgBatch(:,:,1,:);
            G_batch = imgBatch(:,:,2,:);
            B_batch = imgBatch(:,:,3,:);
            
            % Transform each color batch into a 1d Vector
            R_batch = R_batch(:);
            G_batch = G_batch(:);
            B_batch = B_batch(:);
            
            % Create a (batchSize)x(3) matrix
            RGB_matrix = single([R_batch, G_batch, B_batch]);
            
            % Calculate the PCA of the formed RGB matrix, coeff is the
            % eigenvector of the covariance matrix and latent it's
            % eigenvalues of the covariance matrix
            % The PCA automatically preprocess the data...
            [coeff,~,latent] = pca(RGB_matrix); 
            
            eigVec = coeff;
            eigVal = latent;                        
        end
        
        function [newImg] = colorPcaAugmentation(obj, imageIn,eVec,eVal)
            alpha_1 =  randn;
            alpha_2 =  randn;
            alpha_3 =  randn;
            
            alpha_ = [eVal(1)*alpha_1;eVal(2)*alpha_2;eVal(3)*alpha_3]; 
            pca_change = eVec*alpha_;
            
            newImg(:,:,1) = single(imageIn(:,:,1)) + pca_change(1);
            newImg(:,:,2) = single(imageIn(:,:,2)) + pca_change(2);
            newImg(:,:,3) = single(imageIn(:,:,3)) + pca_change(3);
        end
        
        function [barrelImg] = barrelDistortion(obj, imageIn)
            [nrows,ncols] = size(imageIn);
            [xi,yi] = meshgrid(1:ncols,1:nrows);
            resamp = makeresampler('linear','fill');
            % Get image center index
            imid = round(size(imageIn,2)/2); % Find index of middle element
            
            % radial barrel distortion
            xt = xi(:) - imid;
            yt = yi(:) - imid;
            [theta,r] = cart2pol(xt,yt);
            
            a = .0000005; % Try varying the amplitude of the cubic term.
            s = r + a*r.^3;
            [ut,vt] = pol2cart(theta,s);
            u = reshape(ut,size(xi)) + imid;
            v = reshape(vt,size(yi)) + imid;
            tmap_B = cat(3,u,v);
            barrelImg = tformarray(imageIn,[],resamp,[2 1],[1 2],[],tmap_B,.1);
            
            imresize(barrelImg,[nrows,ncols]);
        end
    end
    
end

