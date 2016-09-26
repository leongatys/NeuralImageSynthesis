require 'torch'
require 'nn'

-- Define an nn Module to compute content loss in-place
local MSE, parent = torch.class('nn.MSE', 'nn.Module')

function MSE:__init(targets, weights)
    parent.__init(self)
    self.targets = targets
    self.weights = weights
    self.loss = 0
    self.crit = nn.MSECriterion()
end

function MSE:updateOutput(input)
    self.loss = 0
    if input:nElement() == self.targets[{{1},{},{},{}}]:nElement() then
        for t = 1, self.targets:size()[1] do
            self.loss = self.loss + self.weights[t] * self.crit:forward(input, self.targets[t])
        end
    else
        print('WARNING: Skipping content loss')
    end
    self.output = input
    return self.output
end

function MSE:updateGradInput(input, gradOutput)
    self.gradInput = input.new(#input):fill(0)
    if input:nElement() == self.targets[{{1},{},{},{}}]:nElement() then
        for t = 1, self.targets:size()[1] do
            self.gradInput = self.gradInput + self.crit:backward(input, self.targets[t]):mul(self.weights[t])
        end
    end
    self.gradInput:add(gradOutput)
    return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
    local net = nn.Sequential()
    net:add(nn.View(-1):setNumInputDims(2))
    local concat = nn.ConcatTable()
    concat:add(nn.Identity())
    concat:add(nn.Identity())
    net:add(concat)
    net:add(nn.MM(false, true))
    return net
end

-- Define an nn Module to compute style loss in-place
local GramMSE, parent = torch.class('nn.GramMSE', 'nn.Module')

function GramMSE:__init(targets, weights, guiding_channels)
    parent.__init(self)
    self.targets = targets
    self.weights = weights
    self.guidance = guiding_channels
    self.loss = 0
    self.gram = GramMatrix()
    self.G = nil
    self.crit = nn.MSECriterion()
end

function GramMSE:updateOutput(input)
    local input_chan = input:size()[1]
    if self.guidance then
        input = torch.cat(input, self.guidance, 1)
    end
    self.G = self.gram:forward(input)
    self.G:div(input[{{1},{},{}}]:nElement())
    self.loss = 0
    for t = 1, self.targets:size()[1] do
        self.loss = self.loss + self.weights[t] * self.crit:forward(self.G, self.targets[t])
    end
    if self.guidance then
        input = input[{{1,input_chan},{},{}}]
    end
    self.output = input
    return self.output
end

function GramMSE:updateGradInput(input, gradOutput)
    local input_chan = input:size()[1]
    if self.guidance then
        input = torch.cat(input, self.guidance, 1)
    end
    self.gradInput = input.new(#input):fill(0)
    for t = 1, self.targets:size()[1] do
        local dG = self.crit:backward(self.G, self.targets[t])
        dG:div(input[{{1},{},{}}]:nElement())
        self.gradInput = self.gradInput + self.gram:backward(input, dG):mul(self.weights[t])
    end
    if self.guidance then
        self.gradInput = self.gradInput[{{1,input_chan},{},{}}]
    end
    self.gradInput:add(gradOutput)
    return self.gradInput
end

-- returns layer that computes linear transform in channel dimension
function LinTrans(linear_transform)
    local shape = linear_transform:size()
    local lintrans = nn.SpatialConvolution(shape[2], shape[1], 1, 1, 1, 1, 0, 0)
    lintrans.bias:zero()
    lintrans.weight = linear_transform:typeAs(lintrans.weight)
    return lintrans
end

-- Define an nn Module to compute content loss in-place with linear transform
local LinTransMSE, parent = torch.class('nn.LinTransMSE', 'nn.Module')

function LinTransMSE:__init(targets, weights, linear_transform)
    parent.__init(self)
    self.targets = targets
    self.weights = weights
    self.loss = 0
    self.linear_transform = linear_transform 
    self.trans_input = nil
    self.crit = nn.MSECriterion()
end

function LinTransMSE:updateOutput(input)
    if input[{{1},{1},{},{}}]:nElement() == self.targets[{{1},{1},{},{}}]:nElement() then
        self.trans_input = self.linear_transform:forward(input)
        self.loss = 0
        for t = 1, self.targets:size()[1] do
            self.loss = self.loss + self.weights[t] * self.crit:forward(self.trans_input, self.targets[t])
        end
    else
        print('WARNING: Skipping content loss')
    end
    self.output = input
    return self.output
end

function LinTransMSE:updateGradInput(input, gradOutput)
    if input[{{1},{1},{},{}}]:nElement() == self.targets[{{1},{1},{},{}}]:nElement() then
        self.gradInput = input.new(#input):fill(0)
        local dtrans_input = self.crit:backward(self.trans_input, self.target)
        for t = 1, self.targets:size()[1] do
            self.gradInput = self.gradInput + self.linear_transform:backward(input, dtrans_input):mul(self.weights[t])
        end
    end
    self.gradInput:add(gradOutput)
    return self.gradInput
end

-- Define an nn Module to compute style loss in-place with linear transform
local LinTransGramMSE, parent = torch.class('nn.LinTransGramMSE', 'nn.Module')

function LinTransGramMSE:__init(targets, weights, linear_transform)
    parent.__init(self)
    self.targets = targets
    self.weights = weights
    self.linear_transform = linear_transform 
    self.trans_input = nil
    self.loss = 0
    self.gram = GramMatrix()
    self.G = nil
    self.crit = nn.MSECriterion()
end

function LinTransGramMSE:updateOutput(input)
    self.trans_input = self.linear_transform:forward(input)
    self.G = self.gram:forward(self.trans_input)
    self.G:div(self.trans_input[{{1},{},{}}]:nElement())
    self.loss = 0
    for t = 1, self.targets:size()[1] do
        self.loss = self.loss + self.weights[t] * self.crit:forward(self.G, self.targets[t])
    end
    self.output = input
    return self.output
end

function LinTransGramMSE:updateGradInput(input, gradOutput)
    self.gradInput = input.new(#input):fill(0)
    for t = 1, self.targets:size()[1] do
        local dG = self.crit:backward(self.G, self.targets[t])
        dG:div(self.trans_input[{{1},{},{}}]:nElement())
        local dtrans_input = self.gram:backward(self.trans_input, dG)
        self.gradInput = self.gradInput + self.linear_transform:backward(input, dG):mul(self.weights[t])
    end
    self.gradInput:add(gradOutput)
    return self.gradInput
end

-- Define an nn Module to compute style loss with dilation in-place needs to get input from layer before conv_layer
local GramMSEDilation, parent = torch.class('nn.GramMSEDilation', 'nn.Module')

function GramMSEDilation:__init(targets, weights, conv_layer, dilation_value, guiding_channels)
    parent.__init(self)
    self.targets = targets
    self.weights = weights
    self.guidance = guiding_channels
    self.dilation = nn.Sequential()
    local n_in = conv_layer.nInputPlane 
    local n_out = conv_layer.nOutputPlane 
    local kW, kH, dW, dH =  conv_layer.kH, conv_layer.kW, conv_layer.dH, conv_layer.dW -- kernel size and stride
    local d = dilation_value
    local dl = nn.SpatialDilatedConvolution(n_in, n_out, kW, kH, dW, dH, d, d, d, d)
    dl.weight = conv_layer.weight:clone()
    dl.bias = conv_layer.bias:clone()
    self.dilation:add(dl):add(nn.ReLU())
    self.loss = 0
    self.gram = GramMatrix()
    self.G = nil
    self.crit = nn.MSECriterion()
end

function GramMSEDilation:updateOutput(input)
    local input_chan = input:size()[1]
    local input_dilated = self.dilation:forward(input)
    if self.guidance then
        input_dilated = torch.cat(input_dilated, self.guidance, 1)
    end
    self.G = self.gram:forward(input_dilated)
    self.G:div(input_dilated[{{1},{},{}}]:nElement())
    self.loss = 0
    for t = 1, self.targets:size()[1] do
        self.loss = self.loss + self.weights[t] * self.crit:forward(self.G, self.targets[t])
    end
    self.output = input
    return self.output
end

function GramMSEDilation:updateGradInput(input, gradOutput)
    local input_chan = input:size()[1]
    local input_dilated = self.dilation:forward(input)
    if self.guidance then
        input_dilated = torch.cat(input_dilated, self.guidance, 1)
    end
    local gradInputDilated = input_dilated.new(#input_dilated):fill(0)
    for t = 1, self.targets:size()[1] do
        local dG = self.crit:backward(self.G, self.targets[t])
        dG:div(input_dilated[{{1},{},{}}]:nElement())
        gradInputDilated = gradInputDilated + self.gram:backward(input_dilated, dG):mul(self.weights[t])
    end
    self.gradInput = self.dilation:backward(input, gradInputDilated)
    if self.guidance then
        self.gradInput = self.gradInput[{{1,input_chan},{},{}}]
    end
    self.gradInput:add(gradOutput)
    return self.gradInput
end

-- Define an nn Module to compute style loss in-place and where the loss is only computed from a masked region in the feature map
function GramMatrixMasked(input_chan)
    local net = nn.Sequential()
    net:add(nn.CMulTable())
    local concat = nn.ConcatTable()
    concat:add(nn.View(input_chan,-1))
    concat:add(nn.View(input_chan,-1))
    net:add(concat)
    net:add(nn.MM(false, true))
    return net
end

local GramMSEMasked, parent = torch.class('nn.GramMSEMasked', 'nn.Module')

function GramMSEMasked:__init(targets, weights, masks)
    parent.__init(self)
    self.targets = targets
    self.weights = weights
    self.masks = masks
    self.loss = 0
    self.gram = {}
    for t = 1, self.targets:size()[1] do
        self.gram[t] = GramMatrixMasked(targets:size()[2])
    end
    self.G = {}
    self.crit = nn.MSECriterion()
end

function GramMSEMasked:updateOutput(input)
    local input_chan = input:size()[1]
    self.loss = 0
    for t = 1, self.targets:size()[1] do
        if self.masks[t]:sum() > 0 then
            self.G[t] = self.gram[t]:forward({input, self.masks[t]:repeatTensor(input_chan,1,1)})
            self.G[t]:div(self.masks[t]:sum())
            self.loss = self.loss + self.weights[t] * self.crit:forward(self.G[t], self.targets[t])
        end
    end
    self.output = input
    return self.output
end

function GramMSEMasked:updateGradInput(input, gradOutput)
    local input_chan = input:size()[1]
    self.gradInput = input.new(#input):fill(0)
    for t = 1, self.targets:size()[1] do
        local dG = nil
        if self.masks[t]:sum() > 0 then
            dG = self.crit:backward(self.G[t], self.targets[t])
            dG:div(self.masks[t]:sum())
            self.gradInput = self.gradInput + self.gram[t]:backward({input, self.masks[t]:repeatTensor(input_chan,1,1)}, dG)[1]:mul(self.weights[t])
        end
    end
    self.gradInput:add(gradOutput)
    return self.gradInput
end

-- Computes mean feature maps
function MeanFM()
    local net = nn.Sequential()
    net:add(nn.View(-1):setNumInputDims(2))
    net:add(nn.Mean(2))
    return net
end

-- Define an nn Module to predict aesthetics from mean feature maps
local MeanAesth, parent = torch.class('nn.MeanAesth', 'nn.Module')

function MeanAesth:__init(weight, bias, strength)
    parent.__init(self)
    self.net = nn.Sequential()
    self.net:add(MeanFM())
    local LinTrans = nn.Linear(weight:size()[2], weight:size()[1])
    LinTrans.weight = weight
    LinTrans.bias = bias
    self.net:add(LinTrans)
    self.loss = 0
    self.strength = strength
end

function MeanAesth:updateOutput(input)
    self.loss =  -self.strength * self.net:forward(input)
    self.output = input
    return self.output
end

function MeanAesth:updateGradInput(input, gradOutput)
    grad = input.new(1):fill(self.loss):sign()
    self.gradInput = self.net:backward(input, grad:cmul(self.strength))
    self.gradInput:add(gradOutput)
    return self.gradInput
end

local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(weight)
    parent.__init(self)
    self.weight = weight[1]
    self.loss = 0
    self.x_diff = torch.Tensor()
    self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
    self.output = input
    return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()
    local C, H, W = input:size(1), input:size(2), input:size(3)
    self.x_diff:resize(3, H - 1, W - 1)
    self.y_diff:resize(3, H - 1, W - 1)
    self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
    self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
    self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
    self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
    self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
    self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
    self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
    self.gradInput:mul(self.weight)
    self.gradInput:add(gradOutput)
    return self.gradInput
end

local L2Penalty, parent = torch.class('nn.L2Penalty','nn.Module')

--This module acts as an L2 latent state regularizer, adding the 
--[gradOutput] to the gradient of the L2 loss. The [input] is copied to 
--the [output]. 

function L2Penalty:__init(l2weight, sizeAverage, provideOutput)
    parent.__init(self)
    self.l2weight = l2weight 
    self.sizeAverage = sizeAverage or false  
    if provideOutput == nil then
       self.provideOutput = true
    else
       self.provideOutput = provideOutput
    end
end
    
function L2Penalty:updateOutput(input)
    local m = self.l2weight 
    if self.sizeAverage == true then 
      m = m/input:nElement()
    end
    local loss = m*input:norm(2)/2
    self.loss = loss  
    self.output = input 
    return self.output 
end

function L2Penalty:updateGradInput(input, gradOutput)
    local m = self.l2weight 
    if self.sizeAverage == true then 
      m = m/input:nElement() 
    end
    
    self.gradInput:resizeAs(input):copy(input):mul(m)
    
    if self.provideOutput == true then 
        self.gradInput:add(gradOutput)  
    end 

    return self.gradInput 
end
