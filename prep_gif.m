% addpath(genpath(cd))
% init_folder = '/home/gilad/ssd/keras-frcnn-maste/results_img/car5_30';
% end_folder = [init_folder,'_gif'];
% if ~isdir(end_folder)
%    mkdir(end_folder) 
% end
% skip_num = 40;
% d = dir(init_folder);
% d =d(3:end);
% for i = 1:length(d)
skip_num = 40;
% First create the folder B, if necessary.
init_folder = '/home/gilad/ssd/keras-frcnn-master/results_imgs/car5_30';
outputFolder = [init_folder,'_gif'];
if ~exist(outputFolder, 'dir')
	mkdir(outputFolder);
end
% Copy the files over with a new name.
inputFiles = dir( fullfile(init_folder,'*.png') );
fileNames = { inputFiles.name };
for k = skip_num :skip_num:length(inputFiles )
	thisFileName = fileNames{k};
	% Prepare the input filename.
	inputFullFileName = fullfile(init_folder, thisFileName);
	% Prepare the output filename.
	outputBaseFileName = sprintf('%s.png', num2str(k/skip_num));
	outputFullFileName = fullfile(outputFolder, outputBaseFileName);
	% Do the copying and renaming all at once.
	copyfile(inputFullFileName, outputFullFileName);
end