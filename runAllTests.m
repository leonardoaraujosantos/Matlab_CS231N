%% Script to run all tests (Must be called from console)
%  cd /home/leo/work/Matlab_CS231N
% /usr/local/bin/matlab -nodisplay -r runAllTests

import matlab.unittest.TestSuite;
import matlab.unittest.TestRunner;
import matlab.unittest.plugins.TAPPlugin;
import matlab.unittest.plugins.ToFile;

try
    % Remove any previous TAP file, this avoids errors on jenkins
    if exist('testResults.tap','file')
        delete('testResults.tap')
    end    
    
    % Run all Tests from folder tests
    suite = TestSuite.fromFolder('tests');
    % Create a typical runner with text output
    runner = TestRunner.withTextOutput();
    % Add the TAP plugin and direct its output to a file
    tapFile = fullfile(getenv('WORKSPACE'), 'testResults.tap');
    runner.addPlugin(TAPPlugin.producingOriginalFormat(ToFile(tapFile)));
    % Run the tests
    results = runner.run(suite);
    display(results);
catch e
    disp(getReport(e,'extended'));
    % Exit on any error
    exit(1);
end
% Exit on completion
exit;
