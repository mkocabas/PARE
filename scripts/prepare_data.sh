#!/usr/bin/env bash

gdown 1qIq0CBBj-O6wVc9nJXG-JDEtWPzRQ4KC
unzip pare-github-data.zip
mkdir data/dataset_folders
rm pare-github-data.zip

mkdir -p $HOME/.torch/models/
mv data/yolov3.weights $HOME/.torch/models/
