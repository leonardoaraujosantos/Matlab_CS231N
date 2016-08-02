classdef augmentImage < handle
    %AUGMENTIMAGE Summary of this class goes here
    %   Detailed explanation goes here
    
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
        
        function [flipImg] = flip_H_Image(obj,imageIn)
            flipImg = flip(imageIn,2);
        end
        
        function [flipImg] = flip_V_Image(obj,imageIn)
            flipImg = flip(imageIn,1);
        end
        
        function [flipImg] = flip_Color_Image(obj,imageIn)
            flipImg = flip(imageIn,3);
        end
        
        function [cropImg] = randomCrop(obj, imageIn, numCrops)
            [nrows,ncols] = size(imageIn);
            
            % Standard (Alexnet-paper) ratio for crop
            cropSizeRows = nrows * 0.875;
            cropSizeCols = nrows * 0.875;
            
            % We're going to use imcrop which parameter is a rect with
            % format: [xmin ymin width height], here width=cropSizeCols adn
            % height=cropSizeRows.
            % Return the center crop + (random numCrops-1)
            
            cropImg
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

