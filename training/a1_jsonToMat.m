%% Run getANNO.m in matlab to convert the annotation format from json to mat in {...}/dataset/COCO/mat/
close all; clear variables; clc; tic

% Useful information
% This lines can be executed after the code has finished
% numberImages = 82783 (train2014) & 40504 (val2014)
% numberImagesWithPeople = numel(unique(extractfield(jsonAnnotations, 'image_id')))
% numberPeople = numel(extractfield(jsonAnnotations, 'image_id'))

% User-configurable parameters
loadConfigParameters

% Add COCO Matlab API folder (in order to use its API)
addpath([datasetFolder, '/coco/MatlabAPI/']);

% Create folder where results will be saved
mkdir(matFolder)

% COCO options
annTypes = {'instances', 'captions', 'person_keypoints'};
annType = annTypes{3}; % specify dataType/annType

% Converting and saving validation and training JSON data into MAT format
fprintf('Converting and saving JSON into MAT format\n');
for mode = 0:1
    % Load COCO API with desired (validation vs. training) keypoint annotations
    if mode == 0
        dataType = 'val2014';
    else
        dataType = 'train2014';
    end
    fprintf(['Converting ', dataType, '\n']);
    annotationsFile = sprintf([annotationsFolder, '%s_%s.json'], annType, dataType);
    coco = CocoApi(annotationsFile);
    % Load JSON Annotations
    jsonAnnotations = coco.data.annotations;
    % Auxiliary parameters
    previousImageId = -1;
    imageCounter = 0;
    numberAnnotations = numel(jsonAnnotations);
    logEveryXFrames = round(numberAnnotations / 50);
    % Initialize matAnnotations (no memory allocation)
    matAnnotations = [];
%     % Initialize matAnnotations (memory allocation) (slower!!!)
%     numberImagesWithPeople = numel(unique(extractfield(jsonAnnotations, 'image_id')));
%     matAnnotations = struct('image_id', []);
%     matAnnotations(numberImagesWithPeople).image_id = [];
    % JSON to MAT format
    for i = 1:numberAnnotations
        imageId = jsonAnnotations(i).image_id;
        if imageId == previousImageId
            personCounter = personCounter + 1;
        else
            personCounter = 1;
            imageCounter = imageCounter + 1;
        end
        matAnnotations(imageCounter).image_id = imageId;
        matAnnotations(imageCounter).annorect(personCounter).bbox = jsonAnnotations(i).bbox;
        matAnnotations(imageCounter).annorect(personCounter).segmentation = jsonAnnotations(i).segmentation;
        matAnnotations(imageCounter).annorect(personCounter).area = jsonAnnotations(i).area;
        matAnnotations(imageCounter).annorect(personCounter).id = jsonAnnotations(i).id;
        matAnnotations(imageCounter).annorect(personCounter).iscrowd = jsonAnnotations(i).iscrowd;
        matAnnotations(imageCounter).annorect(personCounter).keypoints = jsonAnnotations(i).keypoints;
        matAnnotations(imageCounter).annorect(personCounter).num_keypoints = jsonAnnotations(i).num_keypoints;
        matAnnotations(imageCounter).annorect(personCounter).img_width = coco.loadImgs(imageId).width;
        matAnnotations(imageCounter).annorect(personCounter).img_height = coco.loadImgs(imageId).height;
        % Remember last image id
        previousImageId = imageId;
        % Display progress
        if mod(i, logEveryXFrames) == 0
            fprintf('%d/%d', i, numberAnnotations);
            if mod(i, 6*logEveryXFrames) == 0
                fprintf('\n');
            else
                fprintf('\t');
            end
        end
    end
    fprintf('\nFinished!\n\n');
    % Save MAT format file
    if mode == 0
        coco_val = matAnnotations;
        save([matFolder, 'coco_val.mat'], 'coco_val');
    else
        coco_kpt = matAnnotations;
        save([matFolder, 'coco_kpt.mat'], 'coco_kpt');
    end
end
% Total running time
disp(['Total time getANNO.m: ', int2str(round(toc)), ' seconds.']);
