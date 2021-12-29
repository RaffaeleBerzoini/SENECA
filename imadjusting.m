clc; clear; close all;

input_dir1 = "build/dataset/Train-Input-PNG-1/";
input_dir2 = "build/dataset/Train-Input-PNG-2/";
input_dir3 = "build/dataset/Train-Input-PNG-3/";
input_dir4 = "build/dataset/Train-Input-PNG-4/";
input_dir5 = "build/dataset/Train-Input-PNG-5/";

directories = [input_dir1, input_dir2, input_dir3, input_dir4, input_dir5];

disp("Adjusting images in:");

for directory = directories
    directory %#ok<*NOPTS> 
    myFiles = dir(fullfile(directory,'*.bmp'));
    for i = 1:length(myFiles)
        filename = myFiles(i).name;
        fullFileName = fullfile(myFiles(i).folder, filename);
        slice = imread(fullFileName);
        low_in = stretchlim(slice);
        slice = imadjust(slice, [low_in(1) low_in(2)], [0 1], 2.1);
        imwrite(slice, fullFileName);
    end
end



