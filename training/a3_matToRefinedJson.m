%% a3_matToJson
% This script creates a cleaner JSON than the original one
% E.g. it removes people too small and/or with very few body parts visible
close all; clear variables; clc

% User-configurable parameters
loadConfigParameters
debugVisualize = 0;

% Time measurement
tic

% Add testing paths
addpath('../testing/util');
addpath('../testing/util/jsonlab/');
% addpath('../testing/util/jsonlab-1.5/');

mkdir([datasetFolder, 'json'])
count = 1;
validationCount = 0;

for mode = 0:1
    % Load COCO API with desired (validation vs. training) keypoint annotations
    if mode == 1
        dataType = 'val2014';
        dataset = 'COCO_val';
        load([matFolder, 'coco_val.mat']);
        matAnnotations = coco_val;
    else
        dataType = 'train2014';
        dataset = 'COCO';
        load([matFolder, 'coco_kpt.mat']);
        matAnnotations = coco_kpt;
    end
    fprintf(['Converting ', dataType, '\n']);

    numberAnnotations = numel(matAnnotations);
    logEveryXFrames = round(numberAnnotations / 40);
    % Process each image with people
    for i = 1:numberAnnotations
        % Display progress
        progressDisplay(i, logEveryXFrames, numberAnnotations);

        previousCenters = [];

        % Prepare for validation
        if mode == 1 && i < 2645
%             fprintf('My validation! %d, %d\n', i, validationCount);
            validationCount = validationCount + 1;
            isValidation = 1;
        else
            isValidation = 0;
        end

        % Width & height of current annotated image
        w = matAnnotations(i).annorect.img_width;
        h = matAnnotations(i).annorect.img_height;

        % Process each person on the image
        numerPeople = length(matAnnotations(i).annorect);
        for person = 1:numerPeople
            % Skip person if number parts is too low or segmentation area  too small
            if matAnnotations(i).annorect(person).num_keypoints >= 5 && matAnnotations(i).annorect(person).area >= 32*32
                % Skip person if distance to exiting person is too small
                personCenter = [matAnnotations(i).annorect(person).bbox(1) + matAnnotations(i).annorect(person).bbox(3) / 2, ...
                                matAnnotations(i).annorect(person).bbox(2) + matAnnotations(i).annorect(person).bbox(4) / 2];
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
                    jointAll(count).dataset = dataset;
                    jointAll(count).isValidation = isValidation;

                    % set image path
                    jointAll(count).img_paths = sprintf([dataType, '/COCO_', dataType, '_%012d.jpg'], matAnnotations(i).image_id);
                    jointAll(count).img_width = w;
                    jointAll(count).img_height = h;
                    jointAll(count).objpos = personCenter;
                    jointAll(count).image_id = matAnnotations(i).image_id;
                    jointAll(count).bbox = matAnnotations(i).annorect(person).bbox;
                    jointAll(count).segment_area = matAnnotations(i).annorect(person).area;
                    jointAll(count).num_keypoints = matAnnotations(i).annorect(person).num_keypoints;


                    % In COCO:(1-'nose'	2-'left_eye' 3-'right_eye' 4-'left_ear' 5-'right_ear'
                    %          6-'left_shoulder' 7-'right_shoulder'	8-'left_elbow' 9-'right_elbow' 10-'left_wrist'	
                    %          11-'right_wrist'	12-'left_hip' 13-'right_hip' 14-'left_knee'	15-'right_knee'	
                    %          16-'left_ankle' 17-'right_ankle' )
                    % set part label: joint_all is (np-3-nTrain)
                    % for this very center person
                    keypoints = matAnnotations(i).annorect(person).keypoints;
                    for part = 1:17
                        jointAll(count).joint_self(part, 1) = keypoints(part*3-2);
                        jointAll(count).joint_self(part, 2) = keypoints(part*3-1);

                        if keypoints(part*3) == 2
                            jointAll(count).joint_self(part, 3) = 1;
                        elseif keypoints(part*3) == 1
                            jointAll(count).joint_self(part, 3) = 0;
                        else
                            jointAll(count).joint_self(part, 3) = 2;
                        end
                    end

                    % set scale
                    jointAll(count).scale_provided = matAnnotations(i).annorect(person).bbox(4) / 368;

                    % for other person on the same image
                    countOther = 1;
                    jointAll(count).joint_others = cell(0,0);
                    for otherPerson = 1:numerPeople
                        if otherPerson ~= person && matAnnotations(i).annorect(otherPerson).num_keypoints > 0
                            keypoints = matAnnotations(i).annorect(otherPerson).keypoints;

                            jointAll(count).scale_provided_other(countOther) = matAnnotations(i).annorect(otherPerson).bbox(4) / 368;
                            jointAll(count).objpos_other{countOther} = [...
                                matAnnotations(i).annorect(otherPerson).bbox(1) + matAnnotations(i).annorect(otherPerson).bbox(3)/2, ...
                                matAnnotations(i).annorect(otherPerson).bbox(2) + matAnnotations(i).annorect(otherPerson).bbox(4)/2 ...
                            ];
                            jointAll(count).bbox_other{countOther} = matAnnotations(i).annorect(otherPerson).bbox;
                            jointAll(count).segment_area_other(countOther) = matAnnotations(i).annorect(otherPerson).area;
                            jointAll(count).num_keypoints_other(countOther) = matAnnotations(i).annorect(otherPerson).num_keypoints;

                            % other people
                            for part = 1:17
                                jointAll(count).joint_others{countOther}(part, 1) = keypoints(part*3-2);
                                jointAll(count).joint_others{countOther}(part, 2) = keypoints(part*3-1);

                                if keypoints(part*3) == 2
                                    jointAll(count).joint_others{countOther}(part, 3) = 1;
                                elseif keypoints(part*3) == 1
                                    jointAll(count).joint_others{countOther}(part, 3) = 0;
                                else
                                    jointAll(count).joint_others{countOther}(part, 3) = 2;
                                end

                            end
                            countOther = countOther + 1;
                        end
                    end
                    jointAll(count).annolist_index = i;
                    jointAll(count).people_index = person;
                    jointAll(count).numOtherPeople = length(jointAll(count).joint_others);

                    % Visualize result (debugging purposes)
                    if debugVisualize
                        imshow([datasetFolder, 'images/', jointAll(count).img_paths]);
                        xlim(jointAll(count).img_width * [-0.6, 1.6]);
                        ylim(jointAll(count).img_height * [-0.6, 1.6]);
                        hold on;
                        visiblePart = jointAll(count).joint_self(:,3) == 1;
                        invisiblePart = jointAll(count).joint_self(:,3) == 0;
                        plot(jointAll(count).joint_self(visiblePart, 1), jointAll(count).joint_self(visiblePart,2), 'gx');
                        plot(jointAll(count).joint_self(invisiblePart,1), jointAll(count).joint_self(invisiblePart,2), 'rx');
                        plot(jointAll(count).objpos(1), jointAll(count).objpos(2), 'cs');
                        if ~isempty(jointAll(count).joint_others)
                            for otherPerson = 1:size(jointAll(count).joint_others,2)
                                visiblePart = jointAll(count).joint_others{otherPerson}(:,3) == 1;
                                invisiblePart = jointAll(count).joint_others{otherPerson}(:,3) == 0;
                                plot(jointAll(count).joint_others{otherPerson}(visiblePart,1), jointAll(count).joint_others{otherPerson}(visiblePart,2), 'mx');
                                plot(jointAll(count).joint_others{otherPerson}(invisiblePart,1), jointAll(count).joint_others{otherPerson}(invisiblePart,2), 'cx');
                                plot(jointAll(count).objpos_other{otherPerson}(1), jointAll(count).objpos_other{otherPerson}(2), 'cs');
                            end
                        end
                        rect_size = 2.1*sqrt(jointAll(count).scale_provided) / 1.2;
%                         max(matAnnotations(i).annorect(person).bbox(3), matAnnotations(i).annorect(person).bbox(4))
%                         sqrt(joint_all(count).scale_provided)
                        rectangle('Position',[jointAll(count).objpos(1)-rect_size, ...
                                              jointAll(count).objpos(2)-rect_size, ...
                                              rect_size*2, ...
                                              rect_size*2], ...
                                  'EdgeColor','b')
                        pause;
                    end
                    %previousCenters = [previousCenters; joint_all(count).objpos joint_all(count).scale_provided*368];
                    previousCenters = [previousCenters; ...
                                       jointAll(count).objpos, ...
                                       max(matAnnotations(i).annorect(person).bbox(3), ...
                                       matAnnotations(i).annorect(person).bbox(4))];
                    count = count + 1;
                end
            end
        end
    end
    fprintf('\nFinished!\n\n');
end

% Save JSON file
opt.FileName = [jsonFolder, 'COCO.json'];
opt.FloatFormat = '%.3f';
fprintf('Saving JSON (it might take several minutes or even a few hours...)\n');
savejson('root', jointAll, opt);
fprintf('\nFinished!\n\n');

% Time measurement
toc
