%% Run genCOCOMask.m in matlab to obatin the mask images for unlabeled person.
% You can use 'parfor' in matlab to speed up the code.
close all; clear variables; clc

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
debug_visualize = 1;

% Start parpool if not started
% Option a
gcp;
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
            if exist(maskAllPath, 'file') && exist(maskMissPath, 'file')
                % Mask image exists, but confirm it was successfully generated
                imread(maskAllPath);
                imread(maskMissPath);
                maskNotGenerated = false;
            end
            % Note: it takes ~3 msec to check both images exist, but ~25
            % msec to load the images. If the exist() command is removed,
            % it would speed up when images are present, but it would
            % considerably slow down when no images are present (e.g. 1st
            % run).
        catch
            maskNotGenerated = true;
        end
        % Generate and write mask
        if maskNotGenerated
            %joint_all(count).img_paths = RELEASE(i).image_id;
            image = imread([datasetFolder, imagePath]);
            [h, w, ~] = size(image);
            maskAll = false(h, w);
            maskMiss = false(h, w);
            flag = 0;
            for p = 1:length(matAnnotations(i).annorect)
                % If this person is annotated
                try
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
                % Else
                catch
                    %display([num2str(i) ' ' num2str(p)]);
                    maskCrowd = logical(MaskApi.decode(matAnnotations(i).annorect(p).segmentation));
                    temp = and(maskAll, maskCrowd);
                    maskCrowd = maskCrowd - temp;
                    flag = flag + 1;
                end
            end
            if flag == 1
                maskMiss = not(or(maskMiss, maskCrowd));
                maskAll = or(maskAll, maskCrowd);
            elseif flag == 0
                maskMiss = not(maskMiss);
            else
                error('flag should never be different than 0 and 1!')
            end
            % Writing resulting image masks
            imwrite(maskAll, maskAllPath);
            imwrite(maskMiss, maskMissPath);
%             % Visualization (debugging purposes)
%             if debug_visualize == 1 && flag == 1
%                 [~, fileName, ~] = fileparts(imagePath);
%                 titleBase = [fileName, ' - '];
%                 % maskPersonP (last person individual mask)
%                 figure(1), blendMask(image, maskPersonP, [titleBase, 'maskPersonP']);
%                 % maskAll
%                 figure(2), blendMask(image, maskAll, [titleBase, 'all mask']);
%                 % maskMiss
%                 figure(3), blendMask(image, maskMiss, [titleBase, 'miss mask']);
%                 % maskCrowd
%                 figure(4), blendMask(image, maskCrowd, [titleBase, 'crowd mask']);
%                 % Pause
%                 pause;
%                 close all;
%             elseif flag > 1
%                 display([num2str(i) ' ' num2str(p)]);
%             end
        end
    end
end
