# User configurable paths
## Path parameters
import os
# sCaffeFolder =  '../caffe_train/'
sCaffeFolder =  '/home/gines/devel/caffe_train/'
sLmdbFolder = '../lmdb_trainVal/'
sPretrainedModelPath = '../vgg/VGG_ILSVRC_19_layers.caffemodel'
sTrainingFolder = '../training_results/pose/'
sTrainedModelsFolder = os.path.join(sTrainingFolder, 'model')
# Relative paths to full paths
sCaffeFolder =  os.path.abspath(sCaffeFolder)
sLmdbFolder = os.path.abspath(sLmdbFolder)
sPretrainedModelPath = os.path.abspath(sPretrainedModelPath)
sTrainingFolder = os.path.abspath(sTrainingFolder)
sTrainedModelsFolder = os.path.abspath(sTrainedModelsFolder)

## Algorithm parameters
sNumberKeyPoints = 18;
sNumberKeyPointsPlusBackground = sNumberKeyPoints+1;
sNumberPAFs = 38;
sImageScale = 368;
sLearningRateInit = 2e-5   # 4e-5, 2e-5
sBatchSize = 21 # 10
sNumberTotalParts = 56
sNumberBodyPartsInLmdb = 17
sMaxRotationDegree = 180 # 40
sBatchNorm = 0
sNumberStages = 6
sScaleMin = 0.1 # 0.5
sScaleMax = 1.1

# Things to try:
# 1. Different batch size --> 20
# 2. Different lr --> 1e-2, 1e-3, 1e-4
# 3. Increase scale_min & scale_max
# 4. Increase max rotation degree



import sys
import os
import math
import argparse
import json
from ConfigParser import SafeConfigParser

sys.path.insert(0, os.path.join(sCaffeFolder, 'python'))
import caffe
from caffe import layers as L  # pseudo module using __getattr__ magic to generate protobuf messages
from caffe import params as P  # pseudo module using __getattr__ magic to generate protobuf messages

def setLayersTwoBranches(dataFolder, batchSize, layerName, kernel, stride, numberOutputChannels, labelName, transformParam, deploy=False, batchNorm=0, lrMultDistro=[1,1,1]):
    # it is tricky to produce the deploy prototxt file, as the data input is not from a layer, so we have to create a workaround
    # producing training and testing prototxt files is pretty straightforward
    caffeNet = caffe.NetSpec()
    assert len(layerName) == len(kernel)
    assert len(layerName) == len(stride)
    assert len(layerName) == len(numberOutputChannels)
    numberParts = transformParam['num_parts']

    # Testing mode
    if deploy:
        input = "image"
        dim1 = 1
        dim2 = 3
        dim3 = 1 # sImageScale
        dim4 = 1 # sImageScale
        # make an empty "data" layer so the next layer accepting input will be able to take the correct blob name "data",
        # we will later have to remove this layer from the serialization string, since this is just a placeholder
        caffeNet.image = L.Layer()
    # Training mode - Use lmdb
    else:
        if "lmdb" not in dataFolder:
            if len(labelName)==1:
                caffeNet.data, caffeNet.tops[labelName[0]] = L.HDF5Data(hdf5_data_param=dict(batch_size=batchSize, source=dataFolder), ntop=2)
            elif len(labelName)==2:
                caffeNet.data, caffeNet.tops[labelName[0]], caffeNet.tops[labelName[1]] = L.HDF5Data(hdf5_data_param=dict(batch_size=batchSize, source=dataFolder), ntop=3)
        # produce data definition for deploy net
        else:
            caffeNet.data, caffeNet.tops['label'] = L.CPMData(data_param=dict(backend=1, source=dataFolder, batch_size=batchSize), 
                                                              cpm_transform_param=transformParam, ntop=2)
            caffeNet.tops[labelName[2]], caffeNet.tops[labelName[3]], caffeNet.tops[labelName[4]], caffeNet.tops[labelName[5]] = L.Slice(caffeNet.label, slice_param=dict(axis=1, slice_point=[sNumberPAFs, numberParts+1, numberParts+sNumberPAFs+1]), ntop=4)
            caffeNet.tops[labelName[0]] = L.Eltwise(caffeNet.tops[labelName[2]], caffeNet.tops[labelName[4]], operation=P.Eltwise.PROD)
            caffeNet.tops[labelName[1]] = L.Eltwise(caffeNet.tops[labelName[3]], caffeNet.tops[labelName[5]], operation=P.Eltwise.PROD)

        # something special before everything
        caffeNet.image, caffeNet.center_map = L.Slice(caffeNet.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
        caffeNet.silence2 = L.Silence(caffeNet.center_map, ntop=0)
        #caffeNet.pool_center_lower = L.Pooling(caffeNet.center_map, kernel_size=9, stride=8, pool=P.Pooling.AVE)

    # just follow arrays..CPCPCPCPCCCC....
    lastLayer = ['image', 'image']
    stage = 1
    convCounter = 1
    poolCounter = 1
    dropCounter = 1
    localCounter = 1
    state = 'image' # can be image or fuse
    sharePoint = 0

    for l in range(0, len(layerName)):
        if layerName[l] == 'V': #pretrained VGG layers
            conv_name = 'conv%d_%d' % (poolCounter, localCounter)
            lr_m = lrMultDistro[0]
            caffeNet.tops[conv_name] = L.Convolution(caffeNet.tops[lastLayer[0]], kernel_size=kernel[l],
                                                     num_output=numberOutputChannels[l], pad=int(math.floor(kernel[l]/2)),
                                                     param=[dict(lr_mult=lr_m, decay_mult=1), dict(lr_mult=lr_m*2, decay_mult=0)],
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     bias_filler=dict(type='constant'))
            lastLayer[0] = conv_name
            lastLayer[1] = conv_name
            print '%s\tch=%d\t%.1f' % (lastLayer[0], numberOutputChannels[l], lr_m)
            ReLUname = 'relu%d_%d' % (poolCounter, localCounter)
            caffeNet.tops[ReLUname] = L.ReLU(caffeNet.tops[lastLayer[0]], in_place=True)
            localCounter += 1
            print ReLUname
        if layerName[l] == 'B':
            poolCounter += 1
            localCounter = 1
        if layerName[l] == 'C':
            if state == 'image':
                #conv_name = 'conv%d_stage%d' % (convCounter, stage)
                conv_name = 'conv%d_%d_CPM' % (poolCounter, localCounter) # no image state in subsequent stages
                if stage == 1:
                    lr_m = lrMultDistro[1]
                else:
                    lr_m = lrMultDistro[1]
            else: # fuse
                conv_name = 'Mconv%d_stage%d' % (convCounter, stage)
                lr_m = lrMultDistro[2]
                convCounter += 1
            #if stage == 1:
            #    lr_m = 1
            #else:
            #    lr_m = lr_sub
            caffeNet.tops[conv_name] = L.Convolution(caffeNet.tops[lastLayer[0]], kernel_size=kernel[l],
                                                     num_output=numberOutputChannels[l], pad=int(math.floor(kernel[l]/2)),
                                                     param=[dict(lr_mult=lr_m, decay_mult=1), dict(lr_mult=lr_m*2, decay_mult=0)],
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     bias_filler=dict(type='constant'))
            lastLayer[0] = conv_name
            lastLayer[1] = conv_name
            print '%s\tch=%d\t%.1f' % (lastLayer[0], numberOutputChannels[l], lr_m)

            if layerName[l+1] != 'L':
                if state == 'image':
                    # Uncommenting this crashes the program
                    # if batchNorm == 1:
                    #     batchNormName = 'bn%d_stage%d' % (convCounter, stage)
                    #     caffeNet.tops[batchNormName] = L.BatchNorm(caffeNet.tops[lastLayer[0]], 
                    #                                          param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
                    #                                          #scale_filler=dict(type='constant', value=1), shift_filler=dict(type='constant', value=0.001))
                    #     lastLayer[0] = batchNormName
                    #ReLUname = 'relu%d_stage%d' % (convCounter, stage)
                    ReLUname = 'relu%d_%d_CPM' % (poolCounter, localCounter)
                    caffeNet.tops[ReLUname] = L.ReLU(caffeNet.tops[lastLayer[0]], in_place=True)
                else:
                    if batchNorm == 1:
                        batchNormName = 'Mbn%d_stage%d' % (convCounter, stage)
                        caffeNet.tops[batchNormName] = L.BatchNorm(caffeNet.tops[lastLayer[0]], 
                                                             param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
                                                             #scale_filler=dict(type='constant', value=1), shift_filler=dict(type='constant', value=0.001))
                        lastLayer[0] = batchNormName
                    ReLUname = 'Mrelu%d_stage%d' % (convCounter, stage)
                    caffeNet.tops[ReLUname] = L.ReLU(caffeNet.tops[lastLayer[0]], in_place=True)
                #lastLayer = ReLUname
                print ReLUname

            #convCounter += 1
            localCounter += 1

        elif layerName[l] == 'C2':
            for level in range(0,2):
                if state == 'image':
                    #conv_name = 'conv%d_stage%d' % (convCounter, stage)
                    conv_name = 'conv%d_%d_CPM_L%d' % (poolCounter, localCounter, level+1) # no image state in subsequent stages
                    if stage == 1:
                        lr_m = lrMultDistro[1]
                    else:
                        lr_m = lrMultDistro[1]
                else: # fuse
                    conv_name = 'Mconv%d_stage%d_L%d' % (convCounter, stage, level+1)
                    lr_m = lrMultDistro[2]
                    #convCounter += 1
                #if stage == 1:
                #    lr_m = 1
                #else:
                #    lr_m = lr_sub
                if layerName[l+1] == 'L2' or layerName[l+1] == 'L3':
                    if level == 0:
                        numberOutputChannels[l] = sNumberPAFs
                    else:
                        numberOutputChannels[l] = sNumberKeyPointsPlusBackground

                caffeNet.tops[conv_name] = L.Convolution(caffeNet.tops[lastLayer[level]], kernel_size=kernel[l],
                                                  num_output=numberOutputChannels[l], pad=int(math.floor(kernel[l]/2)),
                                                  param=[dict(lr_mult=lr_m, decay_mult=1), dict(lr_mult=lr_m*2, decay_mult=0)],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant'))
                lastLayer[level] = conv_name
                print '%s\tch=%d\t%.1f' % (lastLayer[level], numberOutputChannels[l], lr_m)

                if layerName[l+1] != 'L2' and layerName[l+1] != 'L3':
                    if state == 'image':
                        if batchNorm == 1:
                            batchNormName = 'bn%d_stage%d_L%d' % (convCounter, stage, level+1)
                            caffeNet.tops[batchNormName] = L.BatchNorm(caffeNet.tops[lastLayer[level]], 
                                                                 param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
                                                                 #scale_filler=dict(type='constant', value=1), shift_filler=dict(type='constant', value=0.001))
                            lastLayer[level] = batchNormName
                        #ReLUname = 'relu%d_stage%d' % (convCounter, stage)
                        ReLUname = 'relu%d_%d_CPM_L%d' % (poolCounter, localCounter, level+1)
                        caffeNet.tops[ReLUname] = L.ReLU(caffeNet.tops[lastLayer[level]], in_place=True)
                    else:
                        if batchNorm == 1:
                            batchNormName = 'Mbn%d_stage%d_L%d' % (convCounter, stage, level+1)
                            caffeNet.tops[batchNormName] = L.BatchNorm(caffeNet.tops[lastLayer[level]], 
                                                                 param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
                                                                 #scale_filler=dict(type='constant', value=1), shift_filler=dict(type='constant', value=0.001))
                            lastLayer[level] = batchNormName
                        ReLUname = 'Mrelu%d_stage%d_L%d' % (convCounter, stage, level+1)
                        caffeNet.tops[ReLUname] = L.ReLU(caffeNet.tops[lastLayer[level]], in_place=True)
                    print ReLUname

            convCounter += 1
            localCounter += 1
            

        elif layerName[l] == 'P': # Pooling
            caffeNet.tops['pool%d_stage%d' % (poolCounter, stage)] = L.Pooling(caffeNet.tops[lastLayer[0]], kernel_size=kernel[l], stride=stride[l], pool=P.Pooling.MAX)
            lastLayer[0] = 'pool%d_stage%d' % (poolCounter, stage)
            poolCounter += 1
            localCounter = 1
            convCounter += 1
            print lastLayer[0]

        elif layerName[l] == 'L':
            # Loss: caffeNet.loss layer is only in training and testing nets, but not in deploy net.
            if deploy == False and "lmdb" not in dataFolder:
                caffeNet.tops['map_vec_stage%d' % stage] = L.Flatten(caffeNet.tops[lastLayer[0]])
                caffeNet.tops['loss_stage%d' % stage] = L.EuclideanLoss(caffeNet.tops['map_vec_stage%d' % stage], caffeNet.tops[labelName[1]])
            elif deploy == False:
                level = 1
                name = 'weight_stage%d' % stage
                caffeNet.tops[name] = L.Eltwise(caffeNet.tops[lastLayer[level]], caffeNet.tops[labelName[(level+2)]], operation=P.Eltwise.PROD)
                caffeNet.tops['loss_stage%d' % stage] = L.EuclideanLoss(caffeNet.tops[name], caffeNet.tops[labelName[level]])
                
            print 'loss %d' % stage
            stage += 1
            convCounter = 1
            poolCounter = 1
            dropCounter = 1
            localCounter = 1
            state = 'image'

        elif layerName[l] == 'L2':
            # Loss: caffeNet.loss layer is only in training and testing nets, but not in deploy net.
            weight = [lrMultDistro[3],1];
            # print lrMultDistro[3]
            for level in range(0,2):
                if deploy == False and "lmdb" not in dataFolder:
                    caffeNet.tops['map_vec_stage%d_L%d' % (stage, level+1)] = L.Flatten(caffeNet.tops[lastLayer[level]])
                    caffeNet.tops['loss_stage%d_L%d' % (stage, level+1)] = L.EuclideanLoss(caffeNet.tops['map_vec_stage%d' % stage], caffeNet.tops[labelName[level]], loss_weight=weight[level])
                elif deploy == False:
                    name = 'weight_stage%d_L%d' % (stage, level+1)
                    caffeNet.tops[name] = L.Eltwise(caffeNet.tops[lastLayer[level]], caffeNet.tops[labelName[(level+2)]], operation=P.Eltwise.PROD)
                    caffeNet.tops['loss_stage%d_L%d' % (stage, level+1)] = L.EuclideanLoss(caffeNet.tops[name], caffeNet.tops[labelName[level]], loss_weight=weight[level])

                print 'loss %d level %d' % (stage, level+1)
            
            stage += 1
            #last_connect = lastLayer
            #lastLayer = 'image'
            convCounter = 1
            poolCounter = 1
            dropCounter = 1
            localCounter = 1
            state = 'image'

        elif layerName[l] == 'L3':
            # Loss: caffeNet.loss layer is only in training and testing nets, but not in deploy net.
            weight = [lrMultDistro[3],1];
            # print lrMultDistro[3]
            if deploy == False:
                level = 0
                caffeNet.tops['loss_stage%d_L%d' % (stage, level+1)] = L.Euclidean2Loss(caffeNet.tops[lastLayer[level]], caffeNet.tops[labelName[level]], caffeNet.tops[labelName[2]], loss_weight=weight[level])
                print 'loss %d level %d' % (stage, level+1)
                level = 1
                caffeNet.tops['loss_stage%d_L%d' % (stage, level+1)] = L.EuclideanLoss(caffeNet.tops[lastLayer[level]], caffeNet.tops[labelName[level]], loss_weight=weight[level])
                print 'loss %d level %d' % (stage, level+1)
            
            stage += 1
            #last_connect = lastLayer
            #lastLayer = 'image'
            convCounter = 1
            poolCounter = 1
            dropCounter = 1
            localCounter = 1
            state = 'image'

        elif layerName[l] == 'D':
            if deploy == False:
                caffeNet.tops['drop%d_stage%d' % (dropCounter, stage)] = L.Dropout(caffeNet.tops[lastLayer[0]], in_place=True, dropout_param=dict(dropout_ratio=0.5))
                dropCounter += 1
        elif layerName[l] == '@':
            #if not sharePoint:
            #    sharePoint = lastLayer
            caffeNet.tops['concat_stage%d' % stage] = L.Concat(caffeNet.tops[lastLayer[0]], caffeNet.tops[lastLayer[1]], caffeNet.tops[sharePoint], concat_param=dict(axis=1))
            
            localCounter = 1
            state = 'fuse'
            lastLayer[0] = 'concat_stage%d' % stage
            lastLayer[1] = 'concat_stage%d' % stage
            print lastLayer
        elif layerName[l] == '$':
            sharePoint = lastLayer[0]
            poolCounter += 1
            localCounter = 1
            print 'share'

    # Return result
    if deploy:
        caffeNet.tops['net_output'] = L.Concat(caffeNet.tops[lastLayer[0]], caffeNet.tops[lastLayer[1]], concat_param=dict(axis=1))
        deployInit = 'input: {}\n\
input_dim: {} # This value will be defined at runtime\n\
input_dim: {}\n\
input_dim: {} # This value will be defined at runtime\n\
input_dim: {} # This value will be defined at runtime\n'.format('"' + input + '"', dim1, dim2, dim3, dim4)
        # assemble the input header with the net layers string.  remove the first placeholder layer from the net string.
        layerText = 'layer {'
        return deployInit + layerText + layerText.join(str(caffeNet.to_proto()).split(layerText)[2:])
    else:
        return str(caffeNet.to_proto())



def writePrototxts(dataFolder, trainingFolder, batchSize, layerName, kernel, stride, numberOutputChannels, transformParam,
                   learningRateInit, trainedModelsFolder, labelName='label_1st', batchNorm=0, lrMultDistro=[1,1,1]):
    # pose_training.prototxt - Training prototxt
    stringToWrite = setLayersTwoBranches(dataFolder, batchSize, layerName, kernel, stride, numberOutputChannels, labelName, transformParam, deploy=False, batchNorm=batchNorm, lrMultDistro=lrMultDistro)
    with open('%s/pose_training.prototxt' % trainingFolder, 'w') as f:
        f.write(stringToWrite)

    # pose_deploy.prototxt - Deployment prototxt
    stringToWrite = setLayersTwoBranches('', 0, layerName, kernel, stride, numberOutputChannels, labelName, transformParam, deploy=True, batchNorm=batchNorm, lrMultDistro=lrMultDistro)
    with open('%s/pose_deploy.prototxt' % trainingFolder, 'w') as f:
        f.write(stringToWrite)

    # solver.prototxt - Solver parameters
    solver_string = getSolverPrototxt(learningRateInit, trainedModelsFolder)
    with open('%s/pose_solver.prototxt' % trainingFolder, "w") as f:
        f.write('%s' % solver_string)

    # train_pose.sh - Training script
    bash_string = getBash()
    with open('%s/train_pose.sh' % trainingFolder, "w") as f:
        f.write('%s' % bash_string)



def getSolverPrototxt(learningRateInit, snapshotFolder):
    string = '# Net Path Location\n\
net: "pose_training.prototxt"\n\
# Testing\n\
# test_iter specifies how many forward passes the test should carry out.\n\
# In the case of MNIST, we have test batch size 100 and 100 test iterations,\n\
# covering the full 10,000 testing images.\n\
#test_iter: 100\n\
# Carry out testing every 500 training iterations.\n\
#test_interval: 500\n\
# Solver Parameters - Base Learning Rate, Momentum and Weight Decay\n\
base_lr: %f\n\
momentum: 0.9\n\
weight_decay: 0.0005\n\
lr_policy: "step"   # The learning rate policy\n\
gamma: 0.333\n\
stepsize: 100000   # Previously: 29166 68053 136106 (previous one)\n\
# Output - Model Saving and Loss Output\n\
display: 20 # Previously: 5   # Display every X iterations\n\
max_iter: 1000000   # Maximum number of iterations, previously: 600000\n\
snapshot: 10000   # Snapshot intermediate results, previously: 2000\n\
snapshot_prefix: "%s/pose"\n\
solver_mode: GPU   # CPU or GPU\n' % (learningRateInit, snapshotFolder)
    return string



def getBash():
    return ('#!/usr/bin/env sh\n' + sCaffeFolder + '/build/tools/caffe train --solver=pose_solver.prototxt --gpu=$1 \
--weights=' + sPretrainedModelPath + ' 2>&1 | tee ./training_log.txt\n')



if __name__ == "__main__":
    sLearningRateMultDistro = [1.0, 1.0, 4.0, 1.0]
    transformParam = dict(stride=8, crop_size_x=sImageScale, crop_size_y=sImageScale,
                          target_dist=0.6, scale_prob=1, scale_min=sScaleMin, scale_max=sScaleMax,
                          max_rotate_degree=sMaxRotationDegree, center_perterb_max=40, do_clahe=False,
                          visualize=False, np_in_lmdb=sNumberBodyPartsInLmdb, num_parts=sNumberTotalParts)
    if not os.path.exists(sTrainingFolder):
        os.makedirs(sTrainingFolder)

    # # Original/fast versions
    # # First stage             ----------------------- VGG 19 ----------------------- --------------------------------------------------- CPM ---------------------------------------------------
    # layerName               = ['V','V','P'] * 2  +  ['V'] * 4 + ['P']  +  ['V'] * 2 + ['C'] * 2     + ['$'] + ['C2'] * 3 + ['C2'] * 2                       + ['L2']
    # kernel                  = [ 3,  3,  2 ] * 2  +  [ 3 ] * 4 + [ 2 ]  +  [ 3 ] * 2 + [ 3 ] * 2     + [ 0 ] + [ 3 ] * 3  + [ 1 ] * 2                        + [ 0 ]
    # # numberOutputChannels    = [64]*3 + [128]* 3  +  [256] * 4 + [256]  +  [512] * 2 + [256] + [128] + [ 0 ] + [128] * 3  + [512] + [sNumberTotalParts*2]    + [ 0 ] 
    # # numberOutputChannels    = [64]*3 + [128]* 3  +  [256] * 4 + [256]  +  [512] * 2 + [256] + [128] + [ 0 ] + [128] * 3  + [256] + [sNumberTotalParts*2]    + [ 0 ] # Super-fast version 1/2
    # numberOutputChannels    = [64]*3 + [128]* 3  +  [256] * 4 + [256]  +  [512] * 2 + [256] + [128] + [ 0 ] + [128] * 3  + [128] + [sNumberTotalParts*2]    + [ 0 ] # Super-fast version 2/2
    # stride                  = [ 1 , 1,  2 ] * 2  +  [ 1 ] * 4 + [ 2 ]  +  [ 1 ] * 2 + [ 1 ] * 2     + [ 0 ] + [ 1 ] * 3  + [ 1 ] * 2                        + [ 0 ]

    # Super-fast version
    # First stage             ----------------------- VGG 19 ----------------------- --------------------------------------------------- CPM ---------------------------------------------------
    layerName               = ['V','V','P'] * 2  +  ['V'] * 4 + ['P']  +  ['V'] * 2 + ['C'] * 2     + ['$'] + ['C2'] * 2 + ['C2']                   + ['L2']
    kernel                  = [ 3,  3,  2 ] * 2  +  [ 3 ] * 4 + [ 2 ]  +  [ 3 ] * 2 + [ 3 ] * 2     + [ 0 ] + [ 3 ] * 2  + [ 1 ]                    + [ 0 ]
    numberOutputChannels    = [64]*3 + [128]* 3  +  [256] * 4 + [256]  +  [512] * 2 + [256] + [128] + [ 0 ] + [128] * 2  + [sNumberTotalParts*2]    + [ 0 ]
    stride                  = [ 1 , 1,  2 ] * 2  +  [ 1 ] * 4 + [ 2 ]  +  [ 1 ] * 2 + [ 1 ] * 2     + [ 0 ] + [ 1 ] * 2  + [ 1 ]                    + [ 0 ]

    # Stages 2-sNumberStages   ----------------------------------------- CPM + PAF -----------------------------------------
    nodesPerLayer = 5+2
    for s in range(2, sNumberStages+1):
        layerName               += ['@'] + ['C2'] * nodesPerLayer                               +  ['L2']
        kernel                  += [ 0 ] + [ 7 ] * (nodesPerLayer-2) + [1,1]                    +  [ 0 ]
        # numberOutputChannels    += [ 0 ] + [128] * (nodesPerLayer-1) + [sNumberTotalParts*2]    +  [ 0 ] # Original CPM + PAF
        numberOutputChannels    += [ 0 ] + [64] * (nodesPerLayer-1) + [sNumberTotalParts*2]     +  [ 0 ] # (Super-)Fast version
        # numberOutputChannels    += [ 0 ] + [32] * (nodesPerLayer-1) + [sNumberTotalParts*2]     +  [ 0 ] # Super-fast version
        stride                  += [ 0 ] + [ 1 ] * nodesPerLayer                                +  [ 0 ]

    # Create folders where saving
    if not os.path.exists(sTrainingFolder):
        os.makedirs(sTrainingFolder)
    if not os.path.exists(sTrainedModelsFolder): # for storing Caffe models
        os.makedirs(sTrainedModelsFolder)

    labelName = ['label_vec', 'label_heat', 'vec_weight', 'heat_weight', 'vec_temp', 'heat_temp']
    writePrototxts(sLmdbFolder, sTrainingFolder, sBatchSize, layerName, kernel, stride, numberOutputChannels, transformParam, sLearningRateInit, sTrainedModelsFolder, labelName, sBatchNorm, sLearningRateMultDistro)

# NOTE - Speeds
    # Original: slow
    # Fast version: +30%
    # Super-fast version: +XX%
