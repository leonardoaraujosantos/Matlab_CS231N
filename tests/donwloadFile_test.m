%% Test 1: Just download a file and delete it
filenameDownloaded = downloadFile('http://google.com/index.html','index.html');
fprintf('Downloaded file %s\n',filenameDownloaded);
delete(filenameDownloaded);