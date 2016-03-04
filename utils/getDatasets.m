%% Get Cifar 10 for matlab
cifarFile = downloadFile('https://drive.google.com/open?id=0B2RH2qnlKMlEZkxVSWN0bDc2WFU','cifar-10-matlab.tar.gz');
movefile(cifarFile, ['datasets', filesep]);

%% Get Tiny Imagenet
TinyImageNetFile = downloadFile('https://drive.google.com/open?id=0B2RH2qnlKMlEZ1BPQ1hja3ZtLUU','tiny-imagenet-100-A.zip');
movefile(cifarFile, ['datasets', filesep]);