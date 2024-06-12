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
    int GetMaxSamplesToUse() const;

    bool KnowEvalsToDo(int xiCount) const;
    int GetInnerLoopEvalsToPerform(int xiCount) const;

    void SaveInteractionLog(std::string filename);

private:
    double innerOptimisationTimeAllocationms;
    int minInnerLoopEvals;

    int maxMeritSamplesToUse;
    std::vector<int> EvalsToPerformPerxiCount;

    void LogInteraction(std::string desc);
    std::vector<std::string> InteractionLog; // for debugging
};

#endif
