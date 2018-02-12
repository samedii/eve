# Eve
The plan is as follows:
1. Prediction model for toy problem
2. Hyper prediction model for toy problem
3. Prediction model for mnist problem
4. Hyper prediction model for mnist problem
5. Hyper-hyper prediction model for toy problem
6. Hyper-hyper prediction model for mnist problem
7. Switch from backprop to evolutionary methods

Use fixed size networks. Only training the parameters, 
not the structure.

Residual structure? Both for hyper network and prediction network.

Consider changing the plan over the course of development.

## Chapter 1 - Prediction model for toy problem
Train model using backprop.

Needed classes:
* Problem
    * ToyProblem
* PredictionModel
    * ToyPredictionModel

## End goal
Changing environment?

Multiple agents, competition? Would create reasonable level of 
difficulty for further improvement and changing environment? A bit
like self-play?


https://github.com/pytorch/examples/blob/master/mnist/main.py

http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html

