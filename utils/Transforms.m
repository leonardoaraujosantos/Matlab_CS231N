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
            
            R_mean = mean2(R);
            G_mean = mean2(G);
            B_mean = mean2(B);
            
            R_std = std2(R);
            G_std = std2(G);
            B_std = std2(B);
            
            % For each channel subtract mean and divide by standard
            % deviation
            normalizedImg(:,:,1) = (double(R) - R_mean) / R_std;
            normalizedImg(:,:,2) = (double(G) - G_mean) / G_std;
            normalizedImg(:,:,3) = (double(B) - B_mean) / B_std;
        end
        
        function [colorJitImg] = colorJittering(obj, imageIn)
            % Channel to jitter (one at a time)
            channel = randi([1 3]);
            
            % Use a random value bigger than 0.4 but less then 1
            randVal = 0.4 + rand();
            if randVal > 1
                randVal = 1;
            end
            switch channel
                case 1
                    colorJitImg = imadjust(imageIn,[0 0 0; randVal 1 1],[]);
                case 2
                    colorJitImg = imadjust(imageIn,[0 0 0; 1 randVal 1],[]);
                case 3
                    colorJitImg = imadjust(imageIn,[0 0 0; 1 1 randVal],[]);
            end
        end
        
        function [rotImages] = addRotation(obj, imageIn)
            % Get 10 random rotations from -8 .. 8
            rotImages = zeros([size(imageIn),10]);
            for idxImages=1:10
                img = imrotate(imageIn,randi([-8 8]),'crop');
                rotImages(:,:,:,idxImages) = img;
            end
        end
        
        function [noiseImg] = addPeperNoise(obj, imageIn)
            noiseImg = imnoise(imageIn,'gaussian', 0, 0.01 * rand());
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

