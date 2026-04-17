This project investigates the performance differences between an flat model architecture and a hierarchical model architecture both employing a resnet 18 model as backbone model trained from scratch

The flat model implements a mulit-label classification pipeline where the model outputs prediction for the plant and disease at the same time

Whereas the hierarchical architecture has 3 layers, the resnet 18 provids a lower dimensional representation of the image say in (batches, dim) e.g. (4, 514), these are then fed in to a plant classifer, which predicts the crop or plant types say apple or banana given the input feature from the resnet model. Then the plant is used to determine which disease head / sub_model the plant has to be submitted to for disease prediction

The hierarchical model reduces the probablity of incorrect disease prediction which is evidences in the test results and benchmarking datasets collected to evaluate the two architectures

One deduction from the experiment is, the flat model may at times show superior performance in prediction of the crop type, but its performance drops drastically in prediction of the disease type

Whereas the hierarchical architecture has prooved to provide consisent performance across the crop and disease prediction. In addition, out of all benchmarking cases, the hierarchical model provides superior performance across all benchmarks compared to the flat architecture
