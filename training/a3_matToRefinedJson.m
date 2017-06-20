%% a3_matToJson
% This script creates a cleaner JSON than the original one
% E.g. it removes people too small and/or with very few body parts visible
close all; clear variables
% clc

% Time measurement
tic

% User-configurable parameters
loadConfigParameters
debugVisualize = 0;

% Add testing paths
addpath('../testing/util');
addpath('../testing/util/jsonlab/');
% addpath('../testing/util/jsonlab-1.5/');

mkdir([sDatasetFolder, 'json'])
counter = 0;

for mode = 0:1
    % Load COCO API with desired (validation vs. training) keypoint annotations
    if mode == 1
        dataType = 'val2014';
        dataset = 'COCO_val';
        load([sMatFolder, 'coco_val.mat']);
        matAnnotations = coco_val;
    else
        dataType = 'train2014';
        dataset = 'COCO';
        load([sMatFolder, 'coco_kpt.mat']);
        matAnnotations = coco_kpt;
    end
    fprintf(['Converting ', dataType, '\n']);

    numberAnnotations = numel(matAnnotations);
    logEveryXFrames = round(numberAnnotations / 40);
    % Process each image with people
    for imageIndex = 1:numberAnnotations
        % Display progress
        progressDisplay(imageIndex, logEveryXFrames, numberAnnotations);
        % Save 
        previousCenters = [];
        % Prepare for validation
        if mode == 1 && imageIndex < 2645
            isValidation = 1;
        else
            isValidation = 0;
        end
        % Image width & height
        w = matAnnotations(imageIndex).annorect.img_width;
        h = matAnnotations(imageIndex).annorect.img_height;
        % Process each person on the image
        numerPeople = length(matAnnotations(imageIndex).annorect);
        for person = 1:numerPeople
            % Skip person if number parts is too low or segmentation area  too small
            if matAnnotations(imageIndex).annorect(person).num_keypoints >= 5 && matAnnotations(imageIndex).annorect(person).area >= 32*32
                % Skip person if distance to exiting person is too small
                personCenter = [matAnnotations(imageIndex).annorect(person).bbox(1) + matAnnotations(imageIndex).annorect(person).bbox(3) / 2, ...
                                matAnnotations(imageIndex).annorect(person).bbox(2) + matAnnotations(imageIndex).annorect(person).bbox(4) / 2];
                addPerson = true;
                for k = 1:size(previousCenters, 1)
                    dist = norm(previousCenters(k, 1:2) - personCenter);
                    if dist < previousCenters(k, 3) * 0.3
                        addPerson = false;
                        break;
                    end
                end
                % Add person
                if addPerson
                    % Increase counter
                    counter = counter + 1;
                    % Fill new person
                    jointAll(counter).dataset = dataset;
                    jointAll(counter).isValidation = isValidation;
                    jointAll(counter).img_paths = sprintf([dataType, '/COCO_', dataType, '_%012d.jpg'], matAnnotations(imageIndex).image_id);
                    jointAll(counter).img_width = w;
                    jointAll(counter).img_height = h;
                    jointAll(counter).objpos = personCenter;
                    jointAll(counter).image_id = matAnnotations(imageIndex).image_id;
                    jointAll(counter).bbox = matAnnotations(imageIndex).annorect(person).bbox;
                    jointAll(counter).segment_area = matAnnotations(imageIndex).annorect(person).area;
                    jointAll(counter).num_keypoints = matAnnotations(imageIndex).annorect(person).num_keypoints;
                    % Reshape keypoints from [1, (sNumberKeyPoints*3)] to [sNumberKeyPoints, 3]
                    jointAll(counter).joint_self = reshapeKeypoints(matAnnotations(imageIndex).annorect(person).keypoints, sNumberKeyPoints);
                    % Set scale
                    jointAll(counter).scale_provided = matAnnotations(imageIndex).annorect(person).bbox(4) / sImageScale;
                    % Add all other people on the same image
                    counterOther = 0;
                    jointAll(counter).joint_others = cell(0,0);
                    for otherPerson = 1:numerPeople
                        % If other person is not original person and it has >= 1 annotated keypoints
                        if otherPerson ~= person && matAnnotations(imageIndex).annorect(otherPerson).num_keypoints > 0
                            % Increase counter
                            counterOther = counterOther + 1;
                            % Fill person
                            jointAll(counter).scale_provided_other(counterOther) = matAnnotations(imageIndex).annorect(otherPerson).bbox(4) / sImageScale;
                            jointAll(counter).objpos_other{counterOther} = [...
                                matAnnotations(imageIndex).annorect(otherPerson).bbox(1) + matAnnotations(imageIndex).annorect(otherPerson).bbox(3)/2, ...
                                matAnnotations(imageIndex).annorect(otherPerson).bbox(2) + matAnnotations(imageIndex).annorect(otherPerson).bbox(4)/2 ...
                            ];
                            jointAll(counter).bbox_other{counterOther} = matAnnotations(imageIndex).annorect(otherPerson).bbox;
                            jointAll(counter).segment_area_other(counterOther) = matAnnotations(imageIndex).annorect(otherPerson).area;
                            jointAll(counter).num_keypoints_other(counterOther) = matAnnotations(imageIndex).annorect(otherPerson).num_keypoints;
                            % Reshape keypoints from [1, (sNumberKeyPoints*3)] to [sNumberKeyPoints, 3]
                            jointAll(counter).joint_others{counterOther} = reshapeKeypoints(matAnnotations(imageIndex).annorect(otherPerson).keypoints, sNumberKeyPoints);
                        end
                    end
                    jointAll(counter).annolist_index = imageIndex;
                    jointAll(counter).people_index = person;
                    jointAll(counter).numOtherPeople = length(jointAll(counter).joint_others);
                    % Update previous centers
                    previousCenters = [previousCenters; ...
                                       jointAll(counter).objpos, ...
                                       max(matAnnotations(imageIndex).annorect(person).bbox(3), ...
                                       matAnnotations(imageIndex).annorect(person).bbox(4))];
                    % Visualize result (debugging purposes)
                    if debugVisualize
                        imshow([sDatasetFolder, 'images/', jointAll(counter).img_paths]);
                        xlim(jointAll(counter).img_width * [-0.6, 1.6]);
                        ylim(jointAll(counter).img_height * [-0.6, 1.6]);
                        hold on;
                        visiblePart = jointAll(counter).joint_self(:,3) == 1;
                        invisiblePart = jointAll(counter).joint_self(:,3) == 0;
                        plot(jointAll(counter).joint_self(visiblePart, 1), jointAll(counter).joint_self(visiblePart,2), 'gx');
                        plot(jointAll(counter).joint_self(invisiblePart,1), jointAll(counter).joint_self(invisiblePart,2), 'rx');
                        plot(jointAll(counter).objpos(1), jointAll(counter).objpos(2), 'cs');
                        if ~isempty(jointAll(counter).joint_others)
                            for otherPerson = 1:size(jointAll(counter).joint_others,2)
                                visiblePart = jointAll(counter).joint_others{otherPerson}(:,3) == 1;
                                invisiblePart = jointAll(counter).joint_others{otherPerson}(:,3) == 0;
                                plot(jointAll(counter).joint_others{otherPerson}(visiblePart,1), jointAll(counter).joint_others{otherPerson}(visiblePart,2), 'mx');
                                plot(jointAll(counter).joint_others{otherPerson}(invisiblePart,1), jointAll(counter).joint_others{otherPerson}(invisiblePart,2), 'cx');
                                plot(jointAll(counter).objpos_other{otherPerson}(1), jointAll(counter).objpos_other{otherPerson}(2), 'cs');
                            end
                        end
                        rectSize = 2.1 * sqrt(jointAll(counter).scale_provided) / 1.2;
%                         max(matAnnotations(i).annorect(person).bbox(3), matAnnotations(i).annorect(person).bbox(4))
%                         sqrt(joint_all(count).scale_provided)
                        rectangle('Position',[jointAll(counter).objpos(1)-rectSize, ...
                                              jointAll(counter).objpos(2)-rectSize, ...
                                              2*rectSize, ...
                                              2*rectSize], ...
                                  'EdgeColor','b')
                        pause;
                    end
                end
            end
        end
    end
    fprintf('\nFinished!\n\n');
end

% Save JSON file
opt.FileName = [sJsonFolder, 'COCO.json'];
opt.FloatFormat = '%.3f';
fprintf('Saving JSON (it might take several minutes or even a few hours...)\n');
savejson('root', jointAll, opt);
fprintf('\nFinished!\n\n');

% Total running time
disp(['Total time a3_matToRefinedJson.m: ', int2str(round(toc)), ' seconds.']);

function [reshapedKeypoints] = reshapeKeypoints(keypoints, sNumberKeyPoints)
    % Reshape keypoints from [1, (sNumberKeyPoints*3)] to [sNumberKeyPoints, 3]
    % In COCO (sNumberKeyPoints = 17):
    %    (1-'nose'	2-'left_eye' 3-'right_eye' 4-'left_ear' 5-'right_ear'
    %    6-'left_shoulder' 7-'right_shoulder'	8-'left_elbow' 9-'right_elbow' 10-'left_wrist'	
    %    11-'right_wrist'	12-'left_hip' 13-'right_hip' 14-'left_knee'	15-'right_knee'	
    %    16-'left_ankle' 17-'right_ankle')
    reshapedKeypoints = reshape(keypoints, [], sNumberKeyPoints)';
    reshapedKeypoints(:, 3) = mod(reshapedKeypoints(:, 3)+2, 3);
end
