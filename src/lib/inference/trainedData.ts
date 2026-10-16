/**
 * The trained toy weights for server code and tests: imports the vendored
 * file and hands it to the toy model (`provideTrainedWeights`). Never import
 * this from a client component; the browser fetches the same file.
 */
import file from "../../../public/data/trained-weights.json";

import { provideTrainedWeights, type TrainedWeightsFile } from "./toyModel";

provideTrainedWeights(file as unknown as TrainedWeightsFile);

/** Optimiser steps behind the weights (Training's default run). */
export const TRAINED_STEPS: number = file.steps;
