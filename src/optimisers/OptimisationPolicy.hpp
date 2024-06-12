#ifndef OPTIMISATIONPOLICY_HPP
#define OPTIMISATIONPOLICY_HPP

#include <vector>
#include <string>

class OptimisationPolicy {

public:
    void SetInnerOptimisationTimeAllocation(double ms);
    void SetMinInnerLoopEvals(int it);

    void Inform(int xiCount, double weightMatrixCreationTimems);
    void Inform(int xiCount, int innerEvals, double tookms);

    bool KnowMaxMeritSamplesToUse() const;
    int GetMaxMeritSamplesToUse() const;

    bool KnowEvalsToDo(int xiCount) const;
    int GetInnerLoopEvalsToPerform(int xiCount) const;

    void SaveInteractionLog(std::string filename);

private:
    double innerOptimisationTimeAllocationms;
    int minInnerLoopEvals = 1;

    std::vector<bool> haveWeightMatrixInfo;
    std::vector<double> weightMatrixCreationTimesms;

    std::vector<bool> haveInnerLoopTimes;
    std::vector<int> numberOfIt;
    std::vector<double> timeForItsms;

    void UpdateMaxSamplePolicy(int xiCount, double innerOptTotalms);
    int maxMeritSamplesToUse = -1;

    std::vector<bool> knowEvalsToPerform;
    std::vector<int> evalsToPerformPerxiCount;
    double redundancy = 0.95; // how much to undershoot time fill by

    void LogInteraction(std::string desc);
    mutable std::vector<std::string> interactionLog; // for debugging
};

#endif
