%% Generate COCO masks
% Not all COCO images are completely labeled. These masks will tell the
% algorithm which body parts are not labeled, so it does not wrongly uses
% them to generate the brackground heat maps 
close all; clear variables; clc

% Note:
% By default, this code uses the 'parfor' loop in order to speed up the code.
% You can manually disable it.

% Useful information
% Number total masks at the end should be:
% (#imagesWithPeopleTrain2014 + #imagesWithPeopleVal2014) * 2 (a maskAll & maskMiss per image)
% I.e.
% numberMasksOnMask2014 = (numel(coco_kpt) + numel(coco_val)) * 2

% User-configurable parameters
loadConfigParameters

% Add COCO Matlab API folder (in order to use its API)
addpath(cocoMatlabApiFolder);

% Load auxiliary functions
% addpath('../testing/util/'); % mat2im
addpath('util/'); % progressBarInit, progressBarUpdate, blendMask

% Create directory to save generated masks
mkdir(maskFolder)

% Debugging: enable to plot images with mask overlapped
debugVisualize = false;
disableWarnings = true;

% Start parpool if not started
% Option a
p = gcp;
p.IdleTimeout = 525600;
if disableWarnings
    warning ('off','all');
end
% Option b
% delete(gcp('nocreate')); parpool('local', 6);

for mode = 0:1
    % Load annotations from MAT file
    if mode == 0
        load([matFolder, 'coco_val.mat']);
        dataType = 'val2014';
        matAnnotations = coco_val;
    else
        load([matFolder, 'coco_kpt.mat']);
        dataType = 'train2014';
        matAnnotations = coco_kpt;
    end
    numberImagesWithPeople = length(matAnnotations);
    % Display progress bar
    progressBarInit();
    % Enable parfor to speed up the code
    parfor i = 1:numberImagesWithPeople
%     for i = 1:numberImagesWithPeople
        % Update progress bar
        progressBarUpdate(i, numberImagesWithPeople);
        % Paths
        imagePath = sprintf(['images/', dataType, '/COCO_', dataType, '_%012d.jpg'], matAnnotations(i).image_id);
        maskAllPath = sprintf([maskFolder, dataType, '_mask_all_%012d.png'], matAnnotations(i).image_id);
        maskMissPath = sprintf([maskFolder, dataType, '_mask_miss_%012d.png'], matAnnotations(i).image_id);
        % If files exist -> skip (so it can be resumed if cancelled)
        maskNotGenerated = true;
        try
            % Note: it takes ~3 msec to check both images exist, but ~25
            % msec to load the images. If the exist() command is removed,
            % it would speed up when images are present, but it would
            % considerably slow down when no images are present (e.g. 1st
            % run).
            if exist(maskAllPath, 'file') && exist(maskMissPath, 'file')
                % Masks exist, but confirm it was successfully generated
                imread(maskAllPath);
                imread(maskMissPath);
                maskNotGenerated = false;
            end
        catch
            maskNotGenerated = true;
        end
        % Generate and write masks
        if maskNotGenerated
            % Generate masks
            image = imread([datasetFolder, imagePath]);
            [h, w, ~] = size(image);
            maskAll = false(h, w);
            maskMiss = false(h, w);
            peopleOnImageI = length(matAnnotations(i).annorect);
            % If image i is not completely segmented for all people
            try
                % Fill maskAll and maskMiss from each person on image i
                for p = 1:peopleOnImageI
                    % Get person individual mask
                    segmentation = matAnnotations(i).annorect(p).segmentation{1};
                    [X,Y] = meshgrid( 1:w, 1:h );
                    maskPersonP = inpolygon(X, Y, segmentation(1:2:end), segmentation(2:2:end));
                    % Fill mask all
                    maskAll = or(maskPersonP, maskAll);
                    % If not annotations, fill mask miss
                    if matAnnotations(i).annorect(p).num_keypoints <= 0
                        maskMiss = or(maskPersonP, maskMiss);
                    end
                end
                maskMiss = not(maskMiss);
            % If image i is not completely segmented for all people
            catch
                assert(p == peopleOnImageI, 'p should be the last element if no annotations are found!');
                maskNoAnnotations = logical(MaskApi.decode(matAnnotations(i).annorect(p).segmentation));
                maskCrowd = maskNoAnnotations - and(maskAll, maskNoAnnotations);
                maskMiss = not(or(maskMiss, maskCrowd));
                maskAll = or(maskAll, maskCrowd);
            end
            % Write masks
            imwrite(maskAll, maskAllPath);
            imwrite(maskMiss, maskMissPath);
            % Visualize masks (debugging purposes)
            if debugVisualize == 1
                [~, fileName, ~] = fileparts(imagePath);
                titleBase = [fileName, ' - '];
                % maskPersonP (last person individual mask)
                figure(1), blendMask(image, maskPersonP, [titleBase, 'maskPersonP']);
                % maskAll
                figure(2), blendMask(image, maskAll, [titleBase, 'all mask']);
                % maskMiss
                figure(3), blendMask(image, maskMiss, [titleBase, 'miss mask']);
                if exist('maskCrowd', 'var')
                    % maskCrowd
                    figure(4), blendMask(image, maskCrowd, [titleBase, 'crowd mask']);
                    % maskCrowd
                    figure(5), blendMask(image, maskNoAnnotations, [titleBase, 'no annotations mask']);
                end
                % Pause
                pause;
            end
        end
    end
end

if disableWarnings
    warning ('on','all');
end
