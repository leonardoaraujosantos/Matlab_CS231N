%% Get Cifar 10 for matlab
% The parameter dl=1 force to download
cifarFile = downloadFile('https://www.dropbox.com/s/97li582js5ai9uh/cifar-10-matlab.tar.gz?dl=1','cifar-10-matlab.tar.gz');
movefile(cifarFile, ['datasets', filesep]); 

%% Get Tiny Imagenet
TinyImageNetFile = downloadFile('https://www.dropbox.com/s/2gdxwjycu4cj1xf/tiny-imagenet-100-A.zip?dl=1','tiny-imagenet-100-A.zip');
movefile(cifarFile, ['datasets', filesep]);