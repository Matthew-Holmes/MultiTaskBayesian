#include "OptimisationPolicy.hpp"

#include <fstream>


void OptimisationPolicy::SetInnerOptimisationTimeAllocation(double ms) {
    innerOptimisationTimeAllocationms = ms;

    std::string log = "Set inner optimisation time allocation to: ";
    log += std::to_string(ms);
    log += " ms";

    interactionLog.push_back(log);
}

void OptimisationPolicy::SetMinInnerLoopEvals(int it) {
    minInnerLoopEvals = it;

    std::string log = "Set minimum inner loop evals to: ";
    log += std::to_string(it);
    
    interactionLog.push_back(log);
}

void OptimisationPolicy::Inform(int xiCount, double matrixCreationTimems) {
    while (xiCount + 1 > (int)weightMatrixCreationTimesms.size()) {
        weightMatrixCreationTimesms.push_back(0.0);
        haveWeightMatrixInfo.push_back(false);
    }

    weightMatrixCreationTimesms[xiCount] = matrixCreationTimems;
    haveWeightMatrixInfo[xiCount] = true;

    std::string log = "Informed matrix creation for ";
    log += std::to_string(xiCount);
    log += " dimensions, took: ";
    log += std::to_string(matrixCreationTimems);
    log += " ms";

    interactionLog.push_back(log);
}

void OptimisationPolicy::Inform(int xiCount, int innerEvals, double ms) {

    std::string log = "Informed ";
    log += std::to_string(innerEvals);
    log += "surrogate model evals in ";
    log += std::to_string(xiCount);
    log += " dimensions, took: ";
    log += std::to_string(ms);
    log += " ms";
    
    while (xiCount + 1 > (int)timeForItsms.size()) {
        numberOfIt.push_back(0);
        timeForItsms.push_back(0.0);
        haveInnerLoopTimes.push_back(false);
        knowEvalsToPerform.push_back(false);
        evalsToPerformPerxiCount.push_back(0);
    }

    if (innerEvals > numberOfIt[xiCount]) {
        // only update if we will be able to form a better average
        numberOfIt[xiCount] = innerEvals;
        timeForItsms[xiCount] = ms;
        haveInnerLoopTimes[xiCount] = true;

        if (haveWeightMatrixInfo[xiCount]) {
            double predTotal = weightMatrixCreationTimesms[xiCount];
            double timePerIt = timeForItsms[xiCount] / numberOfIt[xiCount];
            predTotal += timePerIt * minInnerLoopEvals;
    
            UpdateMaxSamplePolicy(xiCount, predTotal);

            double timeLeft = innerOptimisationTimeAllocationms * redundancy;
            timeLeft -= weightMatrixCreationTimesms[xiCount];
            int evals = timeLeft / timePerIt; // rounds down
            evalsToPerformPerxiCount[xiCount] = evals;            
            
            std::string log2 = "set evals to perform for ";
            log2 += std::to_string(xiCount);
            log2 += "samples, to ";
            log2 += std::to_string(evals);
            interactionLog.push_back(log2);
        }
    }
}

void OptimisationPolicy::InformFullIterationTime(int it, double ms) {
    std::string log = "full iteration number ";
    log += std::to_string(it);
    log += " took ";
    log += std::to_string(ms);
    log += " ms";
    interactionLog.push_back(log);
}

void OptimisationPolicy::UpdateMaxSamplePolicy(int xiCount, double totalms) {

    if (totalms > innerOptimisationTimeAllocationms * redundancy) {
        if (maxMeritSamplesToUse < xiCount) {
            return;
        } else {
            maxMeritSamplesToUse = xiCount - 1;
            std::string log = "set max merit samples to: ";
            log += std::to_string(xiCount - 1);
            interactionLog.push_back(log);
        }
    } else if (maxMeritSamplesToUse < xiCount) 
    {
        // now have info that using xiCount samples is actually fine
        maxMeritSamplesToUse = xiCount;
        std::string log = "reset max merit samples to: ";
        log += std::to_string(xiCount);
        interactionLog.push_back(log);
    }
}

bool OptimisationPolicy::KnowMaxMeritSamplesToUse() const {
    return maxMeritSamplesToUse != -1;
}

int OptimisationPolicy::GetMaxMeritSamplesToUse() const {
    std::string log = "used max sample policy of ";
    log += std::to_string(maxMeritSamplesToUse);
    log += " samples";
    interactionLog.push_back(log);

    return maxMeritSamplesToUse;
}

bool OptimisationPolicy::KnowEvalsToDo(int xiCount) const {
    if (xiCount + 1 > (int)knowEvalsToPerform.size()) {
        return false;
    } else {
         return knowEvalsToPerform[xiCount];
    }
}

int OptimisationPolicy::GetInnerLoopEvalsToPerform(int xiCount) const {
    std::string log = "used existing eval count of: ";
    log += std::to_string(evalsToPerformPerxiCount[xiCount]);
    log += " evals, for ";
    log += std::to_string(xiCount);
    log += " samples";
    interactionLog.push_back(log);

    return evalsToPerformPerxiCount[xiCount]; 
}

void OptimisationPolicy::SaveInteractionLog(std::string filename) {
    std::ofstream outFile(filename);
    for (const auto &log : interactionLog) outFile << log << "\n";
}
